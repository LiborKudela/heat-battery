import inspect
from ..utilities import load_data, save_data, hash_data
import os
from mpi4py import MPI
import time

class CachedGeometryBuilder:
    def __init__(self, func, dir):
        self.func = func
        self.dir = dir
        self.proccesing_list = []

        spec = inspect.getargspec(func).args
        assert 'dir' in spec, "The builder needs 'dir' key argument"
        assert 'name' in spec, "The builder needs 'name' key argument"
        spec_defaults = inspect.getargspec(func).defaults
        self.default_call_data = dict(zip(spec, spec_defaults))

        self.source_hash = hash_data(inspect.getsource(func))
        self.cache_map_file_path = self.dir + '/cache_map_' + self.source_hash
        if os.path.isfile(self.cache_map_file_path):
            self.cache_map = load_data(self.cache_map_file_path)
        else:
            self.cache_map = {}

    def new_call_data(self, update_with):
        "Return new call_data to the 'func'."
        call_data = self.default_call_data.copy()
        call_data.update(update_with)
        return call_data

    def get_cache_files_location(self, call_data):
        "Return calculated files location."
        spec_id = hash_data(call_data)
        dir = os.path.join(self.dir, call_data['name'])
        return dir, spec_id
    
    def update_cache_map_file(self):
        save_data(self.cache_map_file_path, self.cache_map)

    def files_exist(self, dir, spec_id):
        files_exist_list = [
        os.path.isfile(os.path.join(dir, spec_id + '.msh')),
        os.path.isfile(os.path.join(dir, spec_id + '.ad')),
        os.path.isfile(os.path.join(dir, spec_id + '.step')),
        ]
        return files_exist_list

    def __call__(self, disable_cache=False, **kwargs) -> tuple[str, str]:
        dir, spec_id = None, None
        if MPI.COMM_WORLD.rank == 0:
            print("New geometry requested:")
            call_data = self.new_call_data(kwargs)
            dir, spec_id = self.get_cache_files_location(call_data)
            print(f"  spec_id: {spec_id}")
            print(f"  dir:     {dir}")

            if all(self.files_exist(dir, spec_id)) and not disable_cache:
                print("  method: retriving from cache")
            else:
                print("  method: calculating new geometry")
                call_data['dir'] = dir
                call_data['name'] = spec_id
                self.func(**call_data) # produces the files dir/spec_id
                self.cache_map[spec_id] = dir, spec_id
                self.update_cache_map_file() # data for garbage collector
                print("  New succesfully geometry cached")
        
        dir, spec_id = MPI.COMM_WORLD.bcast([dir, spec_id])
        return dir, spec_id
    
    def run_garbage_collector(self):
        """This removes all files and folders that are not produced by current
        version of func (compared by it source code) given at instantiation
        of this class. This is also the only reason why 'self.cache_map'
        variable exists."""

        # remove all unreletted files
        for r, directories, files in os.walk(self.dir):
            for f in files:
                abs_path = os.path.join(r, f)
                spec_id, extension = os.path.splitext(f)
                if self.cache_map.get(spec_id) is None and abs_path != self.cache_map_file_path:
                    print(f"Removing file: {abs_path}")
                    os.remove(abs_path)

        #delete directories that ended up empty after file deletion
        for r, directories, files in os.walk(self.dir):
            if r != self.dir and len(os.listdir(r)) == 0:
                print(f"Removing empty directoryt: {r}")
                os.rmdir(r)