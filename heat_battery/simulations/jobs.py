from mpi4py import MPI
from typing import Callable
import os
import importlib
import socket
import time
import json
import datetime
import pandas as pd

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
    only_rank_0,
    print_rank_0,
    load_data_json,
)
from ..config import get_config_item
SIGNATURE_LENGTH = 20
GROUP_SIGNATURE_LENGTH = 20

class Job:
    TABLE_NAME = 'jobs'
    COLUMNS= {
        'group_name': 'TEXT',
        'group_signature': 'TEXT',
        'group_priority': 'INTEGER',
        'signature': 'TEXT UNIQUE',
        'created_by': 'TEXT',
        'insert_datetime': 'TIMESTAMP WITH TIME ZONE',
        'last_updated': 'TIMESTAMP WITH TIME ZONE',
        'last_checkpoint_date': 'TIMESTAMP WITH TIME ZONE',
        'last_checkpoint_progress': 'FLOAT',
        'checkpoint_data':'BYTEA',
        'remaining_time': 'INTEGER',
        'elapsed_time': 'INTEGER',
        'priority': 'INTEGER',
        'p_inputs': 'JSONB',
        'output': 'JSONB',
        'probe_columns': 'TEXT[]',
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

    def get_remote_node_name(self):
        if self.is_local():
            result = None
        else:
            result = self.project.get_remote_node_name(self['signature'])
        return result

    def set_remote_node_name(
            self, 
            node_name:str|None,
        ):
        if self.is_local():
            return None
        else:
            if node_name is None:
                node_name = 'UNASSIGNED'
            self.project.set_remote_node_name(
                signature=self['signature'],
                active_node_address=node_name,
            )
    
    def get_status(self):
        if self.is_local():
            result = self.data['status']
        else:
            result = self.project.get_status(self['signature'])
        return result
    
    def set_status(self, status:str):
        if self.is_local():
            self.data['status'] = status
        else:
            self.project.set_status(
                signature=self['signature'],
                status=status,
            )

    def get_error_log(self):
        if self.is_local():
            result = self.data['error_log']
        else:
            result = self.project.get_error_log(self['signature'])
        return result

    def set_error_log(self, error_log:str):
        if self.is_local():
            self.data['error_log'] = error_log
        else:
            self.project.set_error_log(
                signature=self['signature'],
                error_log=error_log,
            )

    def update_progress(self, probes):
        self.project.update_progress(
            signature=self['signature'],
            progress=probes.get_value('progress'),
            remaining_time=probes.get_value('t_remain_avg'),
            elapsed_time=probes.get_value('t_cpu'),
        )

    def update_checkpoint(self):
        temp_dir = get_config_item(['local_temp_dir'])
        main_dir = os.path.join(temp_dir, self['group_signature'])
        checkpoint_dir = os.path.join(main_dir, 'checkpoints', f"{self['signature']}")
        self.project.add_checkpoint(
            signature=self['signature'],
            checkpoint_dir=checkpoint_dir,
        )

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
                file_name = row['file_name']
                data = row['data']
                #file_type = row['file_type']
                #signature = row['sha256']

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
        try_load_checkpoint = False
        if 'SCHEDULED' in org_status:
            if org_remote_node_name == 'UNASSIGNED':
                print_rank_0(
                    f"Job {self.data['signature']} has not been assigned to any "
                    "node before, assigning current worker to it..."
                )
                self.set_remote_node_name(self.get_local_worker_id())
            else:
                print_rank_0(
                    f"Job {self.data['signature']} is SCHEDULED but already "
                    f"assigned to different node {org_remote_node_name}!"
                    "Skipping this job..."
                )
                return None
        elif 'FAILED' in org_status:
            try_load_checkpoint = True
            print_rank_0(
                f"Job {self.data['signature']} has FAILED previously, "
                "assigning current worker to it to try again..."
            )
            self.set_remote_node_name(self.get_local_worker_id())
        elif 'INTERRUPTED' in org_status:
            try_load_checkpoint = True
            print_rank_0(
                f"Job {self.data['signature']} has been INTERRUPTED previously, "
                "assigning current worker to it to try again..."
            )
            self.set_remote_node_name(self.get_local_worker_id())
        elif 'RUNNING' in org_status:
            print_rank_0(
                f"Job {self.data['signature']} is already RUNNING on "
                f"{org_remote_node_name}! This can happen when there is delay "
                "between job being requested and actual call to run(). Skipping this job..."
            )
            return None
        else:
            self.set_remote_node_name(None)
            raise ValueError(
                f"Job {self.data['signature']} has wierd status: {org_status}! "
                f"or wierd remote node name: {org_remote_node_name}! "
                "Please report this bug as it is unexpected!"
            )
        
        self.set_status('RUNNING - FILE PREPARATION')
        self.prepare_files_for_execution()
        MPI.COMM_WORLD.barrier()
        original_cwd = os.getcwd()
        temp_dir = get_config_item(['local_temp_dir'])
        main_dir = os.path.join(temp_dir, self['group_signature'])
        try:
            # import required simulation related code/objects
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
            else:
                print_rank_0("    Mesh already exists in the database...")
            # instantiate simulation in memory
            self.set_status('RUNNING - LOADING SIMULATION')
            checkpoint_dir = os.path.join(main_dir, 'checkpoints', f"{self['signature']}")
            if try_load_checkpoint:
                ckpnt_exists = self.project.check_checkpoint_exist(signature=self['signature'])
                if ckpnt_exists:
                    # download checkpoint from the database
                    self.project.get_checkpoint(
                        signature=self['signature'], 
                        checkpoint_dir=checkpoint_dir,
                    )
                    
                    print_rank_0(
                        f"Checkpoint for job {self['signature']} pulled successfully!"
                    )
                    # read local checkpoint metadata
                    local_metadata = load_data_json(os.path.join(checkpoint_dir, 'metadata.json'))

                    # read remote checkpoint metadata
                    remote_metadata = self.project.get_checkpoint_info(signature=self['signature'])
                    check_keys = ['last_checkpoint_progress', 'last_checkpoint_date']
                    
                    if remote_metadata['progress'] != local_metadata['progress']:
                        print_rank_0(
                            f"WARNING: Checkpoint progress mismatch!\n"
                            f"DB: {remote_metadata['progress']}\n"
                            f"File: {local_metadata['progress']}\n"
                            "This should be investigated as it may indicate a "
                            "bug in the checkpoint upload/download process! "
                            "Please report this bug!"
                        )

                    print_rank_0(f"Checkpoint metadata: {remote_metadata}")
                    load_initial_checkpoint = checkpoint_dir
                else:
                    print_rank_0(
                        f"Checkpoint for job {self['signature']} does not exist "
                        f"in the database even though the job was marked as "
                        f"{org_status}! Skipping checkpoint loading..."
                    )
                    load_initial_checkpoint = None
            else:
                load_initial_checkpoint = None
            sim = sim_class(geometry_dir=geometry_dir, model_name=model_name)
            sim_p = self['p_inputs']['sim_p'] # get runner arguments

            # set probes destination to the database
            sim_p['probe_destinations'] = [
                {
                    'type': 'database',
                    'result_database': self.project.db_name,
                    'project_name': self.project.project_name,
                    'table_name': f"{self['signature']}",
                }
            ]
            if 'probes_callbacks' not in sim_p:
                sim_p['probes_callbacks'] = []
            sim_p['probes_callbacks'].append(self.update_progress)

            # set checkpoint updater
            if 'checkpoint_dt' in sim_p:
                
                os.makedirs(checkpoint_dir, exist_ok=True)
                sim_p['checkpoint_dir'] = checkpoint_dir
                sim_p['load_initial_checkpoint'] = load_initial_checkpoint
                if 'checkpoint_callbacks' not in sim_p:
                    sim_p['checkpoint_callbacks'] = []
                sim_p['checkpoint_callbacks'].append(self.update_checkpoint)

            # run simulation with runner arguments
            self.set_status('RUNNING - SIMULATION')
            runner = getattr(sim, self['runner'])
            runner(**sim_p)

            # set job status to completed
            self.set_status('COMPLETED')
            self.set_remote_node_name(None)

        except Exception as e:
            print_rank_0(f"Job {self.data['signature']} failed with error:")
            import traceback
            error_log = traceback.format_exc()
            print_rank_0(error_log)
            self.set_status('FAILED')
            self.set_remote_node_name(None)
            self.set_error_log(error_log)

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
        group_priority: int,
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
            row = dict(
                file_name=os.path.relpath(dep, start=os.path.dirname(sim_class_file)),
                data=bytes(source, 'utf-8'),
                file_type='source',
                sha256=hash_data(source),
            )
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
            group_signature=p_input['GROUP_SIGNATURE'][:GROUP_SIGNATURE_LENGTH],
            group_priority=group_priority,
            created_by=created_by,
            insert_datetime="UNDEFINED",
            last_updated='UNDEFINED',
            last_checkpoint_date=None,
            last_checkpoint_progress=None,
            checkpoint_data=None,
            remaining_time=0,
            elapsed_time=0,
            signature=p_input['SIGNATURE'][:SIGNATURE_LENGTH],
            priority=p_input['PRIORITY'],
            p_inputs=p_input,
            output='{}',
            probe_columns=[],
            runner=runner,
            required_source_files=[sfr['sha256'] for sfr in source_files_rows],
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
        group_priority: int,
        p_grid: ParameterGrid, 
    ):
    return list(new_jobs_generator(sim_class, mesh_builder, runner, group_name, group_priority, p_grid))

def job_from_legacy_folder(
        legacy_folder:str,
        needs_colums=[]):
    # load sim_p
    unsteady_data = json.load(open(os.path.join(legacy_folder, 'unsteady.meta'), 'r'))
    mesh_data = json.load(open(os.path.join(legacy_folder, 'mesh.meta'), 'r'))
    build_data = json.load(open(os.path.join(legacy_folder, 'build.meta'), 'r'))

    sim_p = build_data.copy()
    sim_p.update(unsteady_data)
    mesh_p = mesh_data.copy()

    res_file_name = os.path.join(legacy_folder, 'unsteady.csv')
    df_res = pd.read_csv(res_file_name)
    try:
        df_res.drop(columns=['t_remain_avg'], inplace=True)
    except:
        pass
    progress = df_res.iloc[-1]['progress']

    data = {
        'group_name': 'Legacy',
        'group_signature': build_data['model_name'].split('-')[0],
        'group_priority': 0,
        'created_by': get_config_item(['user', 'username']),
        'insert_datetime': str(datetime.datetime.now(datetime.timezone.utc)),
        'last_updated': str(datetime.datetime.now(datetime.timezone.utc)),
        'last_checkpoint_date': None,
        'last_checkpoint_progress': None,
        'checkpoint_data': None,
        'signature': legacy_folder.split(os.sep)[-1],
        'remaining_time': None,
        'elapsed_time': None,
        'priority': 0,
        'p_inputs': {
            'sim_p': sim_p,
            'mesh_p': mesh_p,
            'res_file_name': os.path.join(legacy_folder, 'unsteady.csv'),
            'SIGNATURE': legacy_folder.split(os.sep)[-1],
        },
        'output': json.dumps({'key': 'value'}),
        'probe_columns': df_res.columns.tolist(),
        'runner': 'solve_unsteady',
        'required_source_files': [],
        'status': 'COMPLETED',
        'progress': progress,
        'active_node_address': 'UNASSIGNED',
        'error_log': '',
    }
    return Job(data, {}, None)
