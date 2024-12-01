from mpi4py import MPI
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
    only_rank_0,
    print_rank_0,
)
from ..config import get_config_item

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
            group_signature=p_input['GROUP_SIGNATURE'][:10],
            created_by=created_by,
            insert_datetime="UNDEFINED",
            last_updated='UNDEFINED',
            signature=p_input['SIGNATURE'][1:20],
            priority=p_input['PRIORITY'],
            p_inputs=p_input,
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
        p_grid: ParameterGrid, 
    ):
    return list(new_jobs_generator(sim_class, mesh_builder, runner, group_name, p_grid))
