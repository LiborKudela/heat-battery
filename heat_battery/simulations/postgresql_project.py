import pandas as pd
from mpi4py import MPI
import datetime
import json
from typing import Callable
import os
import importlib
import socket

from ..geometry.cached_builder import CachedGeometryBuilder
from .sweep import ParameterGrid
from .simulation_base import Simulation
from ..utilities import (
    compute_function_source_hash_recursive,
    compute_class_source_hash_recursive, 
    combine_hashes,
    get_definition_file_of,
    hash_data,
    resolve_files_dependencies,
)

def only_rank_0(f):
    def wrapper(*args, **kwargs):
        if MPI.COMM_WORLD.rank == 0:
            return f(*args, **kwargs)
    return wrapper

def print_rank_0(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)

import psycopg2
import psycopg2.sql as sql
from psycopg2.extras import execute_values
from ..database.postgresql_connection import DBConnection, get_single_db_connection
from ..config import assert_config_value_set, get_config_item

class Job:
    TABLE_NAME = 'jobs'
    COLUMNS= {
        'group_name': 'TEXT',
        'group_signature': 'TEXT',
        'signature': 'TEXT UNIQUE',
        'created_by': 'TEXT',
        'insert_datetime': 'TIMESTAMP WITH TIME ZONE',
        'last_updated': 'TIMESTAMP WITH TIME ZONE',
        'priority': 'INTEGER',
        'p_inputs': 'JSONB',
        'runner': 'TEXT',
        'required_source_files': 'TEXT[]',
        'status': 'TEXT',
        'progress': 'FLOAT',
        'active_node_address': 'TEXT',
        'error_log': 'TEXT',
    }
    COLUMNS_MAP = {key:i for i, key in enumerate(COLUMNS.keys())}

    def __init__(self, data, source_files_data:dict={}, project=None):
        # INFO: INIT will run only on rank 0, so be cafeful with calling methods
        # that are not MPI safe! It will cause of unsynchronised bcasts and such.
        self.data = data
        self.project = project
        self.source_files_data = source_files_data
        self.custom_worker_id = None

    def __repr__(self):
        return (f"Job(signature={self['signature']}, "
            f"status={self.get_status()}, "
            f"remote={self.is_remote()}, "
            f"progress={self['progress']})"
        )
    
    def __getitem__(self, key):
        return self.data[key]

    def is_local(self):
        return self.project is None
    
    def is_remote(self):
        return not self.is_local()

    def get_remote_project_name(self):
        if self.is_local():
            raise ValueError("Job is not associated with any remote project!")
        return self.project.project_name

    def set_worker_id(self, worker_id:str):
        self.custom_worker_id = worker_id

    def get_local_worker_id(self):
        if self.custom_worker_id is None:
            return f"{MPI.Get_processor_name()}@{self.get_local_rank0_ip_address()}"
        else:
            return self.custom_worker_id
    
    @only_rank_0
    def _get_local_rank0_ip_address(self):
        try:
            # Create a socket connection to an external address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # Google's public DNS server
            ip_address = s.getsockname()[0]  # Get the local IP address
        finally:
            s.close()  # Close the socket
        return ip_address
    
    def get_local_rank0_ip_address(self):
        ip_address = self._get_local_rank0_ip_address()
        return MPI.COMM_WORLD.bcast(ip_address, root=0)
    
    @only_rank_0
    def _get_remote_node_name_query(
            self, 
            conn:psycopg2.extensions.connection=None,
        ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("SELECT active_node_address FROM {} WHERE signature = %s")
            query = query.format(self.project.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (self['signature'],))
            result = cur.fetchone()[0]
        return result

    def get_remote_node_name(self):
        if self.is_local():
            result = None
        else:
            result = self._get_remote_node_name_query()
        return MPI.COMM_WORLD.bcast(result, root=0)

    @only_rank_0
    def _set_remote_node_name_query(
            self, 
            active_node_address:str,
            conn:psycopg2.extensions.connection=None,
        ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("UPDATE {} SET active_node_address = %s WHERE signature = %s")
            query = query.format(self.project.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (active_node_address, self['signature']))
            if commit:
                conn.commit()

    def set_remote_node_name(
            self, 
            node_name:str|None,
        ):
        if self.is_local():
            return None
        else:
            if node_name is None:
                node_name = 'UNASSIGNED'
            self._set_remote_node_name_query(node_name)

    @only_rank_0
    def _get_status_query(
        self,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("SELECT status FROM {} WHERE signature = %s")
            query = query.format(
                self.project.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (self.data['signature'],))
            result = cur.fetchone()[0]
        return result
    
    def get_status(self):
        if MPI.COMM_WORLD.rank == 0:
            if self.is_local():
                result = self.data['status']
            else:
                result = self._get_status_query()
        else:
            result = None
        return MPI.COMM_WORLD.bcast(result, root=0)

    @only_rank_0
    def _set_status_query(
        self, 
        status:str,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("UPDATE {} SET status = %s WHERE signature = %s")
            query = query.format(
                self.project.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (status, self.data['signature']))
            if commit:
                conn.commit()
    
    def set_status(self, status:str):
        if self.is_local():
            self.data['status'] = status
        else:
            self._set_status_query(status)

    @only_rank_0
    def _get_error_log_query(
        self, 
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("SELECT error_log FROM {} WHERE signature = %s")
            query = query.format(self.project.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (self['signature'],))
            result = cur.fetchone()[0]
        return result

    def get_error_log(self):
        if self.is_local():
            result = self.data['error_log']
        else:
            result = self._get_error_log_query()
        return MPI.COMM_WORLD.bcast(result, root=0)
    
    @only_rank_0
    def _set_error_log_query(self, error_log:str, conn:psycopg2.extensions.connection=None):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("UPDATE {} SET error_log = %s WHERE signature = %s")
            query = query.format(self.project.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (error_log, self['signature']))
            if commit:
                conn.commit()

    def set_error_log(self, error_log:str):
        if self.is_local():
            self.data['error_log'] = error_log
        else:
            self._set_error_log_query(error_log)

    @only_rank_0
    def _update_progress_query(
        self, 
        progress:float,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("UPDATE {} SET progress = %s, last_updated = NOW() WHERE signature = %s")
            query = query.format(self.project.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (progress, self['signature']))
            if commit:
                conn.commit()

    def update_progress(self, probes):
        self._update_progress_query(probes.get_value('progress'))

    @only_rank_0
    def prepare_files_for_execution(self, overwrite:bool=False):
        if MPI.COMM_WORLD.rank == 0:
            #print(sys.modules['__main__'].__file__)
            temp_dir = get_config_item(
                ['local_temp_dir'],
                "Local temp directory is not set in the configuration file!",
            )
            main_dir = os.path.join(temp_dir, self['group_signature'])
            if os.path.exists(main_dir) and not overwrite:
                print(f"Directory for group {self['group_signature']} already exists, skipping preparation.")
                return None
            
            print(f"Creating directory for group {self['group_signature']}.")
            print(f"Group_dir: {main_dir}")
            os.makedirs(main_dir, exist_ok=True)
            for row in self.source_files_data:
                file_name, data, file_type, signature = row

                print(f"Writing file {file_name}...", end='')
                file_name = os.path.join(main_dir, file_name)
                dir_name = os.path.dirname(file_name)
                if dir_name != '':  
                    os.makedirs(dir_name, exist_ok=True)
                with open(file_name, 'wb') as f:
                    f.write(data)
                print(" done.")
                # with open(os.path.join(main_dir, 'run_job.py'), 'w') as f:
                #     f.write(self.generate_run_script())

    def generate_local_run_script(
            self, 
            main_dir:str=get_config_item(['local_temp_dir']),
            file_name_format:str='{group_name}_{priority}_run_job.py',
        ):

        code_template = """   
def run():
    from heat_battery.geometry import CachedGeometryBuilder
    import importlib

    sim_class = getattr(importlib.import_module("{sim_module}"), "{sim_class_name}")
    mesh_builder = getattr(importlib.import_module("{mesh_module}"), "{mesh_builder_name}")
    mesh_builder_cached = CachedGeometryBuilder(mesh_builder, "{mesh_cache_dir}")
    mesh_p = {mesh_p}
    sim_p = {sim_p}

    g_dir, model_name = mesh_builder_cached.get_single(**mesh_p)
    sim = sim_class(g_dir, model_name)
    sim.{sim_runner}(**sim_p)
    
if __name__ == "__main__":
    run()
"""
        self.prepare_files_for_execution()
        code = code_template.format(
            sim_module=self['group_signature'] + '.' + os.path.splitext(self['p_inputs']['SIM_CLASS_FILENAME'])[0], 
            sim_class_name=self['p_inputs']['SIM_CLASS_NAME'],
            mesh_module=self['group_signature'] + '.' + os.path.splitext(self['p_inputs']['MESH_BUILDER_FILENAME'])[0], 
            mesh_builder_name=self['p_inputs']['MESH_BUILDER_NAME'],
            mesh_cache_dir=os.path.join(main_dir, 'meshes'),
            sim_runner=self['runner'],
            sim_p=self['p_inputs']['sim_p'],
            mesh_p=self['p_inputs']['mesh_p'],
        )

        if main_dir == "":
            main_dir = os.path.join(self,self['group_signature'])

        with open(os.path.join(main_dir, file_name_format.format(**self.data)), 'w') as f:
            f.write(code)

    def run(self):
        org_status = self.get_status()
        org_remote_node_name = self.get_remote_node_name()
        if org_remote_node_name == 'UNASSIGNED':
            print_rank_0(
                f"Job {self.data['signature']} is not running, assigning "
                "current worker to it..."
            )
            self.set_remote_node_name(self.get_local_worker_id())
        else:
            print_rank_0(
                f"Job {self.data['signature']} is already running on "
                f"{org_remote_node_name}!"
            )
            return None

        if org_status not in ['SCHEDULED', 'FAILED']:
            self.set_remote_node_name(None)
            raise ValueError(
                f"Job {self.data['signature']} has not 'SCHEDULED' or 'FAILED' "
                "status, cannot run it safely!"
            )
        
        self.set_status('RUNNING - FILE PREPARATION')
        self.prepare_files_for_execution()
        MPI.COMM_WORLD.barrier()
        original_cwd = os.getcwd()
        temp_dir = get_config_item(['local_temp_dir'])
        main_dir = os.path.join(temp_dir, self['group_signature'])
        try:
            print_rank_0("Importing modules...")
            self.set_status('RUNNING - IMPORTING MODULES')
            rel_main_dir = ".".join(os.path.relpath(main_dir, start=original_cwd).split(os.sep))
            sim_module = rel_main_dir + '.' + os.path.splitext(self['p_inputs']['SIM_CLASS_FILENAME'])[0]
            print_rank_0("    sim_module:", sim_module)
            sim_class_name = self['p_inputs']['SIM_CLASS_NAME']
            mesh_module = rel_main_dir + '.' + os.path.splitext(self['p_inputs']['MESH_BUILDER_FILENAME'])[0]
            print_rank_0("    mesh_module:", mesh_module)
            mesh_builder_name = self['p_inputs']['MESH_BUILDER_NAME']
            sim_class = getattr(importlib.import_module(sim_module), sim_class_name)
            mesh_builder = getattr(importlib.import_module(mesh_module), mesh_builder_name)
            mesh_p = self['p_inputs']['mesh_p']
            mesh_builder = CachedGeometryBuilder(mesh_builder, os.path.join(main_dir, 'meshes'))

            # resolve mesh files
            self.set_status('RUNNING - PREPARING MESH')
            geometry_dir, model_name = mesh_builder.get_cache_files_location(mesh_p)
            print_rank_0("Looking for mesh files...")
            print_rank_0(f"    Geometry dir: {geometry_dir}")
            print_rank_0(f"    Model name: {model_name}")
            print_rank_0("    Looking for mesh files locally...")
            mesh_present_locally = mesh_builder.files_exist(dir=geometry_dir, spec_id=model_name)
            if not mesh_present_locally:
                print_rank_0("    Mesh files not found...")
                print_rank_0("    Trying to load cached mesh from remote database...")
                self.project.get_meshes(dir=geometry_dir, names=[model_name])
                mesh_loaded_from_db = mesh_builder.files_exist(dir=geometry_dir, spec_id=model_name)
                if not mesh_loaded_from_db:
                    print_rank_0("    Loading mesh from database failed...")
                    print_rank_0("    Trying to build mesh locally...")
                    mesh_builder(**mesh_p) # builds the mesh
            else:
                print_rank_0("    Mesh files found...")
            print_rank_0("    Checking if mesh needs to be reuploaded to the remote database...")
            db_needs_upload = not self.project.check_meshes_exist(names=[model_name])
            if db_needs_upload:
                print_rank_0("    Uploading new mesh to database...")
                self.set_status('RUNNING - UPLOADING MESH')
                self.project.add_meshes(dir=geometry_dir, names=[model_name])

            self.set_status('RUNNING - LOADING SIMULATION')
            sim = sim_class(geometry_dir=geometry_dir, model_name=model_name)

            self.set_status('RUNNING - SIMULATION')
            sim_p = self['p_inputs']['sim_p']
            if 'probes_callbacks' not in sim_p:
                sim_p['probes_callbacks'] = []
            sim_p['probes_callbacks'].append(self.update_progress)
            runner = getattr(sim, self['runner'])
            runner(**sim_p)
            self.set_status('COMPLETED')
            self.set_remote_node_name(None)
        except Exception as e:
            print_rank_0(f"Job {self.data['signature']} failed with error:")
            import traceback
            error_log = traceback.format_exc()
            print_rank_0(error_log)
            self.set_status('FAILED')
            self.set_error_log(error_log)
        finally:
            os.chdir(original_cwd)  

    def get_probes_data(self, as_dataframe:bool=False):
        if self.is_local():
            return None
        else:
            return self.project.get_result_table(signature=self['signature'], as_dataframe=as_dataframe)

def new_jobs_generator(
        sim_class: type[Simulation], 
        mesh_builder: Callable,
        runner: str,
        group_name: str,
        p_grid: ParameterGrid, 
    ):
    assert issubclass(sim_class, Simulation), f"Expected Simulation, got {type(sim_class)}"
    assert isinstance(p_grid, ParameterGrid), f"Expected ParameterGrid, got {type(p_grid)}"
    
    # model (Simulation class)
    # TODO: compute the signature from all dependencies
    sim_class_signature = compute_class_source_hash_recursive(sim_class)
    sim_class_file = get_definition_file_of(sim_class)

    # mesh builder (function that dumps mesh files)
    mesh_builder_signature = compute_function_source_hash_recursive(mesh_builder)
    mesh_builder_file = get_definition_file_of(mesh_builder)

    # mesh file name generator

    p_grid = p_grid.copy()
    p_grid.dict.update({
        'SIM_CLASS_NAME': sim_class.__name__,
        'SIM_CLASS_SIGNATURE': sim_class_signature,
        'SIM_CLASS_FILENAME': os.path.basename(sim_class_file),

        'MESH_BUILDER_NAME': mesh_builder.__name__,
        'MESH_BUILDER_SIGNATURE': mesh_builder_signature,
        'MESH_BUILDER_FILENAME': os.path.basename(mesh_builder_file),

        'GROUP_SIGNATURE': combine_hashes(
            [sim_class_signature, mesh_builder_signature]
        ),
    })

    dependencies = [sim_class_file, mesh_builder_file]
    dependencies += resolve_files_dependencies(
        dependencies, 
        constraint_path=os.path.dirname(sim_class_file),
    )

    # all source files involved in the simulation
    source_files_rows = []
    for dep in dependencies:
        with open(dep, 'r') as f:
            source = f.read()
            row = [
                os.path.relpath(dep, start=os.path.dirname(sim_class_file)),
                bytes(source, 'utf-8'),
                'source',
                hash_data(source),
            ]
            source_files_rows.append(row)
            
    # p_grid.dict.update(
    #     {'DEPENDENCIES': dependencies,
    #      }
    # )
    p_grid.instantiate()
    created_by = get_config_item(['user', 'username'])

    for p_input in p_grid.kde_parameters():

        job_row = dict(
            group_name=group_name,
            group_signature=p_input['GROUP_SIGNATURE'][:10],
            created_by=created_by,
            insert_datetime="UNDEFINED",
            last_updated='UNDEFINED',
            signature=p_input['SIGNATURE'][1:20],
            priority=p_input['PRIORITY'],
            p_inputs=p_input,
            runner=runner,
            required_source_files=[sfr[Project.FILES_COLUMNS_MAP['sha256']] for sfr in source_files_rows],
            status='SCHEDULED',
            progress=0,
            active_node_address='UNASSIGNED',
            error_log='',
        ) 

        if set(job_row.keys()) != set(Job.COLUMNS.keys()):
            raise ValueError(
                "Job data does not match the expected columns! "
                f"{job_row.keys()} vs. expected {Job.COLUMNS.keys()}"
            )

        yield Job(job_row, source_files_rows)

def generate_jobs(
        sim_class: type[Simulation], 
        mesh_builder: Callable,
        runner: str,
        group_name: str,
        p_grid: ParameterGrid, 
    ):
    return list(new_jobs_generator(sim_class, mesh_builder, runner, group_name, p_grid))

@only_rank_0
def _create_database_query(if_exists:str='skip'):
    conn = get_single_db_connection('postgres')
    conn.autocommit = True

    query = sql.SQL(
            "SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = %s)"
        )
    cur = conn.cursor()
    cur.execute(query, (Project.DB_NAME,))
    res = cur.fetchone()[0]
    if res:
        if if_exists == 'skip': 
            print(
                f"Database '{Project.DB_NAME}' already exists, "
                "skipping creation"
            )
            return None # <- stop here
        elif if_exists == 'fail':
            raise ValueError(
                f"Database '{Project.DB_NAME}' already exists, "
                "cannot proceed further!"
            )
        else:
            raise ValueError(
                "Unknown value of if_exists. Expected 'skip' or 'fail', "
                f"got '{if_exists}'."
            )

    print(f"Database '{Project.DB_NAME}' does not exist, creating it")
    query = sql.SQL("CREATE DATABASE {}")
    query = query.format(sql.Identifier(Project.DB_NAME))
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    conn.close()

def create_database(if_exists:str='skip'):
    _create_database_query(if_exists=if_exists)

class Project:
    DB_NAME = get_config_item(['database', 'postgres', 'db_name'])

    FILES_TABLE_NAME = 'files'
    FILES_COLUMNS = {
        'file_name': 'TEXT',
        'data': 'BYTEA',
        'type': 'TEXT',
        'sha256': 'TEXT UNIQUE',
    }
    FILES_COLUMNS_MAP = {key:i for i, key in enumerate(FILES_COLUMNS.keys())}

    def __init__(self, project_name: str, if_exists:str='skip'):
        self.project_name = project_name
        self.ensure_database_exists()
        self.create(if_exists=if_exists)

    def get_jobs_table_sql_identifier(self, project_name:str=None):
        project_name = project_name if project_name is not None else self.project_name
        return sql.Identifier(project_name, Job.TABLE_NAME)

    def get_files_table_sql_identifier(self, project_name:str=None):
        project_name = project_name if project_name is not None else self.project_name
        return sql.Identifier(project_name, Project.FILES_TABLE_NAME)

    def ensure_database_exists(self):
        _create_database_query(if_exists='skip')

    @only_rank_0
    def _exists_query(self, conn:psycopg2.extensions.connection=None):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL(
                "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = %s)"
            )
            cur = conn.cursor()
            cur.execute(query, (self.project_name,))
            res = cur.fetchone()[0]
        return res

    def exists(self):
        return self._exists_query()

    @only_rank_0
    def _drop_query(
        self,
        fail_if_not_exists: bool = False,
        cascade: bool = False,
        conn:psycopg2.extensions.connection = None, 
        ):

        commit = conn is None

        with DBConnection() if commit else conn as conn:

            query = sql.SQL("DROP SCHEMA {} {} {}")
            query = query.format(
                sql.SQL('IF EXISTS') if fail_if_not_exists else sql.SQL(''),
                sql.Identifier(self.project_name),
                sql.SQL('CASCADE') if cascade else sql.SQL(''),
            )
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()

    def drop(
        self,
        fail_if_not_exists: bool = False,
        cascade: bool = False,
    ):
        self._drop_query(fail_if_not_exists=fail_if_not_exists, cascade=cascade)

    @only_rank_0
    def _create_query(
        self,
        conn:psycopg2.extensions.connection = None,
        if_exists: str = 'skip',

        ):

        commit = conn is None
        project_name = self.project_name

        with DBConnection() if commit else conn as conn:
            print(self._exists_query(conn))
            if self._exists_query(conn):
                if if_exists == 'skip':
                    print(f"Schema '{project_name}' already exists, skipping creation")
                    return None
                elif if_exists == 'override':
                    self._drop_query(conn=conn, fail_if_not_exists=False, cascade=True)
                    print(f"Schema '{project_name}' already exists, but was dropped as if_exists = 'override' was set.")
                elif if_exists == 'fail':
                    raise ValueError(f"Schema '{project_name}' already exists, cannot proceed further!")
                else:
                    raise ValueError(f"Unknown value of if_exists. Expected 'skip', 'override' or 'fail', got '{if_exists}'.")

            query = sql.SQL("CREATE SCHEMA IF NOT EXISTS {}")
            query = query.format(sql.Identifier(project_name))
            cur = conn.cursor()
            cur.execute(query)

            column_specs = [
                sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(type))
                for name, type in Job.COLUMNS.items()
            ]
            query = sql.SQL("CREATE TABLE {} ({})")
            query = query.format(
                self.get_jobs_table_sql_identifier(project_name),
                sql.SQL(', ').join(column_specs)
            )
            cur = conn.cursor()
            cur.execute(query)

            column_specs = [
                sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(type))
                for name, type in Project.FILES_COLUMNS.items()
            ]
            query = sql.SQL("CREATE TABLE {} ({})")
            query = query.format(
                self.get_files_table_sql_identifier(project_name),
                sql.SQL(', ').join(column_specs)
            )
            cur = conn.cursor()
            cur.execute(query)

            if commit:
                conn.commit()

    def create(self, if_exists: str = 'skip'):
        self._create_query(if_exists=if_exists)

    @only_rank_0
    def _get_info_query(self, conn:psycopg2.extensions.connection=None):
        with DBConnection() if conn is None else conn as conn:

            query = sql.SQL(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s"
            )

            cur = conn.cursor()
            cur.execute(query, (self.project_name,))
            table_names = cur.fetchall()

            query =sql.SQL("UNION ").join([
                sql.SQL("SELECT COUNT(*) FROM {} ").format(
                    sql.Identifier(self.project_name, name[0])
                )
                for name in table_names
            ])
            print(query.as_string(conn))
            cur.execute(query)
            table_data = cur.fetchall()
            res = {t_name[0]: table_data[0] for t_name, table_data in zip(reversed(table_names), table_data)}
        return res

    def get_info(
            self,
            as_dataframe:bool=False,
            ):
        res = self._get_info_query()
        res = MPI.COMM_WORLD.bcast(res, root=0)
        if as_dataframe:
            res = pd.DataFrame(res, columns=['table_name'])
        return res

    @only_rank_0
    def _reset_all_statuses_query(
        self, 
        new_status:str='SCHEDULED',
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("UPDATE {} SET status = %s")
            query = query.format(
                self.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (new_status,))
            if commit:
                conn.commit()

    def reset_all_statuses(self, new_status:str='SCHEDULED'):
        self._reset_all_statuses_query(new_status=new_status)

    @only_rank_0
    def _add_files_query(
        self, 
        files_rows:list, 
        conn:psycopg2.extensions.connection=None,
        ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL(
                "INSERT INTO {} ({}) VALUES %s ON CONFLICT DO NOTHING"
            ).format(
                self.get_files_table_sql_identifier(),
                sql.SQL(', ').join(
                    sql.Identifier(col) for col in Project.FILES_COLUMNS.keys()
                ),
            )
            cur = conn.cursor()
            execute_values(cur, query, files_rows)

            if commit:
                conn.commit()

    def add_files(self, files_rows:list):
        self._add_files_query(files_rows=files_rows)

    @only_rank_0
    def _check_files_exist_query(self, signatures:list[str], conn):
        with DBConnection() as conn:
            query = sql.SQL("SELECT EXISTS (SELECT 1 FROM {} WHERE signature IN %s)")
            query = query.format(
                self.get_files_table_sql_identifier()
            )
            cur = conn.cursor()
            cur.execute(query, (tuple(signatures), ))
            res = cur.fetchone()[0]
        return res

    def check_files_exist(self, signatures:list[str]):
        res = self._check_files_exist_query(signatures=signatures)
        return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    def _get_files_query(
        self, 
        conn:psycopg2.extensions.connection = None, 
        signatures:list[str] | None = None,
        ):
        with DBConnection() if conn is None else conn as conn:
            if signatures is not None:
                query = sql.SQL("SELECT * FROM {} WHERE sha256 IN %s")
            else:
                query = sql.SQL("SELECT * FROM {}")
            query = query.format(
                self.get_files_table_sql_identifier()
            )
            cur = conn.cursor()
            if signatures is not None:
                cur.execute(query, (tuple(signatures), ))
            else:
                cur.execute(query)
            rows = cur.fetchall()
            rows = [[row[0], row[1].tobytes(), row[2], row[3]] for row in rows]
        return rows
    
    def get_files(
            self, 
            signatures:list[str]=None, 
            as_dataframe:bool=False,
        ):
        rows = self._get_files_query(signatures=signatures)
        rows = MPI.COMM_WORLD.bcast(rows, root=0)
        if as_dataframe:
            return pd.DataFrame(rows, columns=Project.FILES_COLUMNS.keys())
        else:
            return rows
    
    @only_rank_0
    def _add_jobs_query(
            self, 
            jobs:list,
            conn:psycopg2.extensions.connection = None,
            ):

        commit = conn is None
        with DBConnection() if commit else conn as conn:
            page_size = 1000

            # insert jobs
            jobs_query = sql.SQL(
                "INSERT INTO {} ({}) VALUES %s ON CONFLICT DO NOTHING"
            ).format(
                self.get_jobs_table_sql_identifier(),
                sql.SQL(', ').join(
                    sql.Identifier(col) for col in Job.COLUMNS.keys()
                ),
            )
            cur = conn.cursor()
            job_rows = []
            unique_source_files_rows = set()
            for job in jobs:

                # jobs table data copy not to mess with the original data
                cp_data = job.data.copy()

                # insert and last updated datetime
                cp_data['insert_datetime'] = str(datetime.datetime.now(datetime.timezone.utc))
                cp_data['last_updated'] = cp_data['insert_datetime']

                # set remote probe output location
                cp_data['p_inputs']['sim_p']['result_database'] = get_config_item(['database', 'postgres', 'db_name'])
                cp_data['p_inputs']['sim_p']['project_name'] = self.project_name
                cp_data['p_inputs']['sim_p']['database_table'] = f'res_{cp_data["signature"]}'

                # dump p_inputs into database compatible jsonb format
                cp_data['p_inputs'] = json.dumps(cp_data['p_inputs'])
                job_rows.append([cp_data[key] for key in Job.COLUMNS.keys()])

                # files table data
                for row in job.source_files_data:
                    hashable_row = tuple(row)
                    if hashable_row not in unique_source_files_rows:
                        unique_source_files_rows.add(hashable_row)

            execute_values(cur, jobs_query, job_rows, page_size=page_size)
            self._add_files_query(files_rows=list(unique_source_files_rows))

            if commit:
                conn.commit()

    def add_jobs(self, jobs):
        self._add_jobs_query(jobs=jobs)

    def _add_meshes_query(
        self,
        dir:str,
        names:list[str], 
        conn:psycopg2.extensions.connection = None,
    ):

        commit = conn is None
        files_rows = []
        for name in names:
            for ext in ['msh', 'ad', 'step']:
                file_name = ".".join([name, ext])
                file_path = os.path.join(dir, file_name)
                with open(file_path, 'rb') as f:
                    data = f.read()
                    files_rows.append((file_name, data, f'mesh_{ext}', hash_data(data)))
        with DBConnection() if commit else conn as conn:
            self._add_files_query(files_rows=files_rows, conn=conn)

            if commit:
                conn.commit()

    def add_meshes(self, dir:str, names:list[str]):
        self._add_meshes_query(dir=dir, names=names)

    @only_rank_0
    def _get_meshes_query(
        self, 
        dir:str,
        names:list[str], 
        conn:psycopg2.extensions.connection=None,
    ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("SELECT * FROM {} WHERE file_name IN %s")
            query = query.format(
                self.get_files_table_sql_identifier(),
            )
            cur = conn.cursor()

            for name in names:
                files_names = []
                for ext in ['msh', 'ad', 'step']:
                    files_names.append(".".join([name, ext]))
                cur.execute(query, (tuple(files_names), ))
                data_rows = cur.fetchall()
                for data_row in data_rows:
                    with open(os.path.join(dir, data_row[Project.FILES_COLUMNS_MAP['file_name']]), 'wb') as f:
                        f.write(data_row[Project.FILES_COLUMNS_MAP['data']])
    
    def get_meshes(self, names:list[str], dir:str):
        res = self._get_meshes_query(dir=dir, names=names)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    def _check_meshes_exist_query(
        self, 
        names:list[str], 
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("SELECT EXISTS (SELECT 1 FROM {} WHERE file_name IN %s)")
            query = query.format(
                self.get_files_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (tuple(names), ))
            res = cur.fetchone()[0]
        return res

    def check_meshes_exist(self, names:list[str]):
        res = self._check_meshes_exist_query(names=names)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    def _get_jobs_query(
        self, 
        conn:psycopg2.extensions.connection=None, 
        as_dataframe:bool=False,
        ):
        if MPI.COMM_WORLD.rank == 0:
            with DBConnection() if conn is None else conn as conn:
                query = sql.SQL(
                    "SELECT * FROM {}"
                )
                query = query.format(
                    self.get_jobs_table_sql_identifier(),
                    )
                cur = conn.cursor()
                cur.execute(query)
                jobs_rows = cur.fetchall()
            return jobs_rows

    def get_jobs(
            self, 
            as_dataframe:bool=False,
        ):
        jobs_rows = self._get_jobs_query(as_dataframe=as_dataframe)
        jobs_rows = MPI.COMM_WORLD.bcast(jobs_rows, root=0)
        if as_dataframe:
            return pd.DataFrame(jobs_rows, columns=Job.COLUMNS.keys())
        else:
            return [Job(dict(zip(Job.COLUMNS.keys(), row)), [], self) for row in jobs_rows]  

    @only_rank_0
    def _get_next_scheduled_job_query(
        self, 
        conn:psycopg2.extensions.connection=None,
        ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL(
                "SELECT * FROM {} WHERE status = 'SCHEDULED' OR status LIKE 'FAILED%%' ORDER BY priority ASC LIMIT 1"
            )  
            query = query.format(
                self.get_jobs_table_sql_identifier(),
                )
            cur = conn.cursor()
            cur.execute(query)
            row = cur.fetchone()

            required_source_files = row[Job.COLUMNS_MAP['required_source_files']]
            source_files_rows = self._get_files_query(conn=conn, signatures=required_source_files)
        if row is not None:
            return Job(dict(zip(Job.COLUMNS.keys(), row)), source_files_rows, self)

    def get_next_scheduled_job(self) -> Job | None:
        res = self._get_next_scheduled_job_query()
        res = MPI.COMM_WORLD.bcast(res, root=0)
        MPI.COMM_WORLD.barrier()
        return res

    @only_rank_0
    def _get_job_query(
        self, 
        signature:str, 
        conn:psycopg2.extensions.connection=None, 
        ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL(
                "SELECT * FROM {} WHERE signature = %s"
            )
            query = query.format(
                self.get_jobs_table_sql_identifier(self.project_name),
                )
            cur = conn.cursor()
            cur.execute(query, (signature,))
            row = cur.fetchone()

            required_source_files = row[Job.COLUMNS_MAP['required_source_files']]
            source_files_rows = self._get_files_query(conn=conn, signatures=required_source_files)
        if row is not None:
            return Job(dict(zip(Job.COLUMNS.keys(), row)), source_files_rows, self)
        
    def get_job(self, signature:str):
        res = self._get_job_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    def _reset_uncompleted_jobs_status_query(
        self,
        new_status:str='SCHEDULED',
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("UPDATE {} SET status = %s WHERE status != 'COMPLETED' AND last_updated < NOW() - INTERVAL '10 minutes'")
            query = query.format(
                self.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (new_status, ))
            if commit:
                conn.commit()

    def reset_uncompleted_jobs_status(self, new_status:str='SCHEDULED'):
        self._reset_uncompleted_jobs_status_query(new_status=new_status)

    # @only_rank_0
    # def _get_interupted_jobs_query(self, conn:psycopg2.extensions.connection=None):
    #     with DBConnection() if conn is None else conn as conn:
    #         query = sql.SQL("SELECT * FROM {} WHERE status != 'COMPLETED' AND last_updated < NOW() - INTERVAL '10 minutes'")
    #         query = query.format(
    #             self.get_jobs_table_sql_identifier(),
    #         )
    #         cur = conn.cursor()
    #         cur.execute(query)
    #         res_rows = cur.fetchall()
    #         return [Job(dict(zip(Job.COLUMNS.keys(), row)), [], self) for row in res_rows]   

    # def get_interupted_jobs(self):
    #     res = self._get_interupted_jobs_query()
    #     return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    def _get_result_table_names_query(self):
        with DBConnection() as conn:
            query = sql.SQL("SELECT table_name FROM information_schema.tables WHERE table_schema = %s AND table_name LIKE 'res\_%%'")
            cur = conn.cursor()
            cur.execute(query, (self.project_name, ))
            res = cur.fetchall()
        return res
    
    def get_result_table_names(self):
        res = self._get_result_table_names_query()
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    def _get_result_table_query(
        self, 
        signature:str,
        skip_first_n_rows:int=None,
        read_first_n_rows:int=None,
        read_last_n_rows:int=None,
    ):
        with DBConnection() as conn:

            # check if table exists
            table_exists_query = sql.SQL("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s)")
            cur = conn.cursor()
            cur.execute(table_exists_query, (self.project_name, f'res_{signature}'))
            table_exists = cur.fetchone()[0]
            if not table_exists:
                print(f"Table 'res_{signature}' does not exist!")
                return None, None

            # get columns
            cols_query = sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s")
            cur = conn.cursor()
            cur.execute(cols_query, (self.project_name, f'res_{signature}'))
            cols = [col[0] for col in cur.fetchall()]

            # get rows
            query = sql.SQL("SELECT * FROM {} ")
            query = query.format(
                sql.Identifier(self.project_name, f'res_{signature}'),
            )
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
        return cols, rows
    
    def get_result_table(
            self, 
            signature:str, 
            as_dataframe:bool=False,
        ):
        res = self._get_result_table_query(signature=signature)
        res = MPI.COMM_WORLD.bcast(res, root=0)
        cols, rows = res
        if as_dataframe:
            return pd.DataFrame(columns=cols, data=rows)
        else:
            return res

    @only_rank_0
    def _clean_files_query(
        self, 
        conn:psycopg2.extensions.connection=None, 
        ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL(
                "SELECT array_agg(uq) FROM "
                "(SELECT DISTINCT unnest(required_source_files) as uq FROM {})"
                "subquery"
            )
            query = query.format(
                self.get_jobs_table_sql_identifier(self.project_name),
                )
            cur =  conn.cursor()
            cur.execute(query)
            res = cur.fetchone()[0]
        return res
    
    def clean_files(self):
        res = self._clean_files_query()
        return MPI.COMM_WORLD.bcast(res, root=0)
 
    def __repr__(self):
        return f"Project(name={self.project_name})"

def _list_remote_projects_query(conn:psycopg2.extensions.connection=None):
    with DBConnection() if conn is None else conn as conn:
        query = sql.SQL(
            "SELECT nspname FROM pg_namespace "
            "WHERE nspname NOT IN ('pg_toast', 'pg_catalog', 'public', 'information_schema')"
        )
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        res = [Project(row[0]) for row in rows]
        return res
    
def list_remote_projects():
    res = _list_remote_projects_query()
    return MPI.COMM_WORLD.bcast(res, root=0)
