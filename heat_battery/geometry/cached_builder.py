import inspect
from ..utilities import hash_data
import os
from mpi4py import MPI

class CachedGeometryBuilder:
    def __init__(self, func, dir, project=None):
        self.func = func
        self.dir = dir
        self.project = project
        self.proccesing_list = []

        spec = inspect.getargspec(func).args
        assert 'dir' in spec, "The builder needs 'dir' key argument"
        assert 'name' in spec, "The builder needs 'name' key argument"
        spec_defaults = inspect.getargspec(func).defaults
        self.default_call_data = dict(zip(spec, spec_defaults))

        self.source_hash = hash_data(inspect.getsource(func))

    def new_call_data(self, update_with):
        "Return new call_data to the 'func'."
        call_data = self.default_call_data.copy()
        call_data.update(update_with)
        return call_data

    def get_cache_files_location(self, kwargs):
        "Return calculated files location."
        # ignore these entries
        full_call_data = self.new_call_data(kwargs)
        name_chunks = (
            self.source_hash, 
            hash_data((full_call_data, self.source_hash)),
        )
        spec_id = '-'.join(name_chunks)
        dir = os.path.join(self.dir, full_call_data['name'])
        return dir, spec_id

    def files_exist(self, dir, spec_id):
        files_exist_list = [
        os.path.isfile(os.path.join(dir, spec_id + '.msh')),
        os.path.isfile(os.path.join(dir, spec_id + '.ad')),
        os.path.isfile(os.path.join(dir, spec_id + '.step')),
        ]
        return all(files_exist_list)

    def get_single(self, disable_cache=False, return_was_from_cache=False, only_rank=0,**kwargs):
        was_from_cache = False
        dir, spec_id = self.get_cache_files_location(kwargs)

        if self.files_exist(dir, spec_id) and not disable_cache:
            was_from_cache = True
        else:
            kwargs['dir'] = dir
            kwargs['name'] = spec_id
            self.func(**kwargs) # produces the files dir/spec_id
        if return_was_from_cache:
            return dir, spec_id, was_from_cache
        else:
            return dir, spec_id
        
    def get_multiple(self, kwargs_list, disable_cache=False, parallel=True, error_on_fail=True, return_counts=False):
        n_success = 0
        n_failed = 0
        n_cached = 0
        n_generated = 0
        dirs = []
        spec_ids = []
        if parallel:
            rank_offset = 0
            for i, mesh_p in enumerate(kwargs_list):
                if (MPI.COMM_WORLD.rank + rank_offset) == i:
                    rank_offset += MPI.COMM_WORLD.size
                    try:
                        dir, spec_id, was_from_cache = self.get_single(
                            **mesh_p, 
                            disable_cache=disable_cache, 
                            return_was_from_cache=True
                            )
                        n_cached += int(was_from_cache)
                        n_generated += 1 - int(was_from_cache)
                        n_success += 1
                        dirs.append(dir)
                        spec_ids.append(spec_id)
                    except Exception as e:
                        if error_on_fail:
                            raise Exception(e)
                        else:
                            print(f"Build failed with: {e}")
                            print(f"  call data: {mesh_p}")
                            n_failed += 1
            MPI.COMM_WORLD.barrier()
            n_success = MPI.COMM_WORLD.allreduce(n_success, op=MPI.SUM)
            n_failed = MPI.COMM_WORLD.allreduce(n_failed, op=MPI.SUM)
            n_cached = MPI.COMM_WORLD.allreduce(n_cached, op=MPI.SUM)
            n_generated = MPI.COMM_WORLD.allreduce(n_generated, op=MPI.SUM)
            dirs = MPI.COMM_WORLD.allgather(dirs)
            dirs = [item for sublist in dirs for item in sublist]
            spec_ids = MPI.COMM_WORLD.allgather(spec_ids)
            spec_ids = [item for sublist in spec_ids for item in sublist]
        elif not parallel:
            for mesh_p in kwargs_list:
                if MPI.COMM_WORLD.rank == 0:
                    try:
                        dir,spec_id, was_from_cache = self.get_single(**mesh_p, return_was_from_cache=True)
                        n_cached += int(was_from_cache)
                        n_generated += 1 - int(was_from_cache)
                        n_success += 1
                        dirs.append(dir)
                        spec_ids.append(spec_id)
                    except Exception as e:
                        if error_on_fail:
                            raise Exception(e)
                        else:
                            print(f"Build failed with: {e}")
                            print(f"  call data: {mesh_p}")
                            n_failed += 1
        else:
            raise Exception("'parallel' arg must be boolean.")
        counts = (n_success, n_failed, n_cached, n_generated)
        if return_counts:
            return (dirs, spec_ids, counts)
        else:
            return (dirs, spec_ids)

    def __call__(
        self, 
        disable_cache=False, 
        return_was_from_cache=False, 
        **kwargs
    ) -> tuple[str, str, bool]:

        dir, spec_id, was_from_cache = None, None, None
        if MPI.COMM_WORLD.rank == 0:
            dir, spec_id, was_from_cache = self.get_single(
                disable_cache=disable_cache, 
                return_was_from_cache=True, 
                **kwargs
            )
        dir, spec_id, was_from_cache = MPI.COMM_WORLD.bcast([dir, spec_id, was_from_cache])
        if return_was_from_cache:
            return dir, spec_id, was_from_cache
        else:
            return dir, spec_id
    
    def cleanup_unused_files(self):
        """This removes all files and folders that are not produced by current
        version of func (compared by it source code) given at instantiation
        of this class."""

        if MPI.COMM_WORLD.rank == 0:

            # remove all unreletted files
            for r, directories, files in os.walk(self.dir):
                for f in files:
                    abs_path = os.path.join(r, f)
                    spec_id, extension = os.path.splitext(f)
                    splited_spec = spec_id.split('-')
                    if len(splited_spec) < 2 or splited_spec[0] != self.source_hash:
                        print(f"Removing file: {abs_path}")
                        os.remove(abs_path)

            #delete directories that ended up empty after file deletion
            for r, directories, files in os.walk(self.dir):
                if r != self.dir and len(os.listdir(r)) == 0:
                    print(f"Removing empty directoryt: {r}")
                    os.rmdir(r)
        MPI.COMM_WORLD.Barrier()

        