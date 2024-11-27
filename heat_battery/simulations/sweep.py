from ..utilities import (
    compute_function_source_hash_recursive, 
    compute_class_source_hash_recursive, 
    combine_hashes,
    get_definition_file_of,
    hash_data,
)
from ..geometry import CachedGeometryBuilder
import os
from mpi4py import MPI
import numpy as np
from scipy.spatial import distance
import hashlib
import json

class ParameterList:
    """
    A list of parameters that can be used to generate a one dimension of the parameter space.
    """
    def __init__(self, l):
        self.l = l

    def __add__(self, other):
        if isinstance(other, ParameterList):
            return ParameterList(self.l+other.l)
        else:
            raise Exception(f"Cannot add {type(other)} to ParameterList")

    def __len__(self):
        return len(self.l)
    
    def __getitem__(self, i):
        return self.l[i]

class ParameterEvaluation:
    """ 
    A class that is used to evaluate a parameter value from a formula that
    expresses the value as some combination of other parameters from the grid.
    The evaluation is defered until the whole parameter set is instantiated. 
    Objects can be passed in to the defered scope via defered_scope argument.
    The formulas can access the values of other parameters from the current parameter
    set by using the 'SELF' keyword (capitals intended). Keyword 'SELF' evaluates
    to the current parameter set (dictionary) at the top most level. Dictionary
    SELF always contains two special values:
      - 'SIGNATURE' which gives an unique hash (string) of the given parameter
        set.
      - 'PRIORITY' which gives a priority (int) of the given parameter set in
        the grid.

    SimJobController automatically adds the following special values to the SELF
    dictionary:
      - 'SIM_CLASS_NAME' which gives a name (string) of the simulation class.
      - 'SIM_CLASS_SIGNATURE' which gives an unique hash (string) of the
        simulation class.
      - 'MESH_BUILDER_SIGNATURE' which gives an unique hash (string) of the mesh
        builder.
      - 'GROUP_SIGNATURE' which gives an unique hash (string) for a combination
        of sim_class and mesh_builder.

    These special keywords are useful when we need to generate some unique
    identifications. (such as unique file names for each parameter set). No two
    parameter sets will have the same 'SIGNATURE' as it uses cryptographically
    secure hashing (sha256) algorithm and no two parameter sets generated via
    given ParameterGrid will have the same 'PRIORITY' as it defines the prefered
    order of evaluation of the parameter space to span widest variance first.
    When using ParameterGrid inside SimJobController, the string 'SIGNATURE'
    will change to a unique value with a change in model source code or/and mesh
    builder source code.

    Args:
        formula: A string that is the formula that evaluates to the parameter
        value. eval_as_code: A boolean that indicates if the formula should be
        evaluated as python code or as a string.

    Example:
        >>> pg = ParameterGrid(dict(    
                arg_1 = 1.0,
                arg_2 = ParameterList([1.0, 2.0]),
                arg_3 = ParameterEvaluation("SELF['arg_1'] + SELF['arg_2']", eval_as_code=True),
                arg_4 = ParameterEvaluation("SELF['arg_1'] + SELF['arg_2']", eval_as_code=False),
                arg_5 = ParameterEvaluation("{SELF['arg_1']} + {SELF['arg_2']}", eval_as_code=False),
                arg_6 = ParameterEvaluation("{SEFL['PRIORITY']}/{SELF['SIGNATURE']}", eval_as_code=False),
            ))

        >>> p_sets = list(pg.kde_parameters())
        >>> print(p_sets[0])
        {'arg_1': 1.0, 
        'arg_2': 1.0, 
        'arg_3': 2.0, 
        'arg_4': "SELF['arg_1'] + SELF['arg_2']", 
        'arg_5': '1.0 + 1.0', 
        'arg_6': '0/ad15bf63a3de7383537e334adfd47e7f1da2bc61b6098f7b309d6137b230a474'}

        >>> print(p_sets[1])
        {'arg_1': 1.0, 
        'arg_2': 2.0, 
        'arg_3': 3.0, 
        'arg_4': "SELF['arg_1'] + SELF['arg_2']", 
        'arg_5': '1.0 + 2.0', 
        'arg_6': '1/0c5cc459bfe23e05f00f9571d3101e2a58dd2e054536b12f798271655581645e'}
    """
    def __init__(self, formula, deferred_scope={}, eval_as_code=False):
        self.formula = formula
        self.deferred_scope = deferred_scope
        self.eval_as_code = eval_as_code

    def evaluate(self, locals): 
        scope = locals.copy()
        scope.update(self.deferred_scope)
        fstr = eval(f'f"""{self.formula}"""', scope)
        if self.eval_as_code:
            return eval(fstr, scope)
        else:
            return fstr
        
class NoNumericalEffect:
    """
    A placeholder value that is used to indicate that a parameter has no
    numerical effect on the numerical simulation results. It can be used to
    control things such as verbosity level of terminal output. This value will
    not affect 'SIGNATURE' hash of the particular parameter set which means that
    change of NoNumericalEffect value will not invalidate already generated data.

    Args:
        value: The value of the NoNumericalEffect parameter.

    Example:
        >>> pg1 = ParameterGrid(dict(
                arg_0 = ParameterList([1.0, 2.0]),
                no_effect_value = NoNumericalEffect('some_value'),
                signature = ParameterEvaluation("{SIGNATURE}"),
            ))
        >>> pg2 = ParameterGrid(dict(
                arg_0 = ParameterList([1.0, 2.0]),
                no_effect_value = NoNumericalEffect('diferent_value'),
                signature = ParameterEvaluation("{SIGNATURE}"),
            ))
        >>> l1 = list(pg1.kde_parameters())
        >>> l2 = list(pg2.kde_parameters())

        >>> print(l1[0]['no_effect_value'] == l2[0]['no_effect_value'])
        False
        >>> print(l1[0]['signature'] == l2[0]['signature'])
        True
    """

    def __init__(self, value):
        self.value = value

class ParameterGrid():
    """
    A class that is used to generate a grid of parameter sets.

    Args:
        dict: A dictionary that contains the parameters and their values.

    Example:
        >>> pg = ParameterGrid(dict(    
                arg_1 = 1.0,
                arg_2 = ParameterList([1.0, 2.0]),
                arg_3 = ParameterEvaluation("SELF['arg_1'] + 1", eval_as_code=True),
                arg_4 = NoNumericalEffect("some_value"),
            ))
    """
    def __init__(self, dict):
        self.dict = dict

    def instantiate(self):
        self.names = self.dict.keys()
        self.values = self.dict.values()
        self.param_lists = self.get_param_lists() # potentialy recursive call
        self.param_paths = self.get_param_paths() # potentialy recursive call
        self.lengths = [len(pl) for pl in self.param_lists]
        self.name_map = list(self.names)
        self.n = self.count()
        self.salts = []
       
    def get_param_lists(self):
        pl = []
        for key, value in self.dict.items():
            if isinstance(value, ParameterList):
                pl.append(value)
            elif isinstance(value, ParameterGrid):
                pl += value.get_param_lists()
            else:
                constant = ParameterList([value])
                pl.append(constant)
        return pl
    
    def get_param_paths(self):
        paths = []
        for key, value in self.dict.items():
            if isinstance(value, ParameterList):
                paths.append([key])
            elif isinstance(value, ParameterGrid):
                child_keys = value.get_param_paths()
                for ck in child_keys:
                    if isinstance(ck, list):
                        paths.append([key] + ck)
                    else:
                        paths.append([key, ck])
            else:
                paths.append([key])
        return paths
    
    def get_grid_structure_signature(self):
        m = hashlib.sha256()
        for pp in self.param_paths:
            m.update(json.dumps(pp).encode('UTF-8'))
        return m.hexdigest()

    def get_parameter_set_signature(self, values):
        m = hashlib.sha256()
        for val, pp in zip(values, self.param_paths):
            if isinstance(val, NoNumericalEffect):
                continue
            elif isinstance(val, ParameterEvaluation):
                #TODO: defered scope missing, ad it to the hash
                d_str = json.dumps((val.formula, val.eval_as_code, pp))
            else:
                d_str = json.dumps((val, pp))
            m.update(d_str.encode('UTF-8'))

        for salt in self.salts:
            m.update(salt.encode('UTF-8'))
        return m.hexdigest()
    
    def populate_param_dict(self, values, priority):
        nested_dict = {}
        PE_paths = []
        
        # Walk through all paths and set the values
        for keys, value in zip(self.param_paths, values):
            current_dict = nested_dict
            
            # Go through the path to the end 
            for key in keys[:-1]:
                if key not in current_dict:
                    # Create a new sub path if key doesn't exist
                    current_dict[key] = {}
                current_dict = current_dict[key]
            
            # We are at most outer branch, set the value of the key
            if isinstance(value, ParameterEvaluation):
                PE_paths.append(keys)
            elif isinstance(value, NoNumericalEffect):
                value = value.value
            current_dict[keys[-1]] = value
        
        # special keywords so particular param set can be uniquely identified
        nested_dict['SIGNATURE'] = self.get_parameter_set_signature(values)
        nested_dict['PRIORITY'] = priority
        
        # Evaluate values of each ParameterEvaluation inside the tree
        # TODO: check for cyclic definitions, assert non-cyclic
        
        for keys in PE_paths:
            # Go through the path to the end and find the ParameterEvaluation
            current_dict = nested_dict 
            for key in keys[:-1]:
                current_dict = current_dict[key]
            pe = current_dict[keys[-1]]
            current_dict[keys[-1]] = pe.evaluate({'SELF': nested_dict})
        
        return nested_dict

    def kde_parameters(self):
        for priority, idxs in enumerate(self.kde_indexes()):
            values = [pl[idx] for idx, pl in zip(idxs, self.param_lists)]
            yield self.populate_param_dict(values, priority)
    
    def kde_indexes(self):
        indexes = np.array(self.ordered_index_array())
        n = len(indexes)
        flags = np.zeros(indexes.shape[0], dtype=bool)
        distances = distance.cdist(indexes, indexes, metric='euclidean')
        kernels = 1/(distances**2/2 + 1e-17)
        densities = np.full(n, 0.0, dtype=float)
        next_point = 0
        flags[0] = True
        yield indexes[0]

        # For each iteration select point that has lowest density
        for j in range(1, n):
            densities += kernels[:, next_point]
            densities[next_point] = np.inf
            next_point = np.argmin(densities)
            flags[next_point] = True
            yield indexes[next_point]

    def ordered_index_array(self):
        out = []
        for i in range(len(self)):
            out.append(self.index_combination(i))
        return out

    def index_combination(self, i):
        c = []
        for length in reversed(self.lengths):
            max_index = length - 1 
            index = i % (max_index + 1)
            c.append(index)
            i //= (max_index + 1)
        c.reverse()
        return c
    
    def count(self):
        n = 1
        for pl in self.lengths:
            n *= pl
        return n

    def __len__(self):
        return self.n
    
    def get_unique_by_param_path(self, param_path):
        values = []
        for p_inputs in self.kde_parameters():
            for key in param_path:
                p_inputs = p_inputs[key]
            if p_inputs not in values:
                values.append(p_inputs)
        return values
    
    def copy(self):
        return ParameterGrid(self.dict.copy())
  
class JobGenerator:
    def __init__(self, sim_class, mesh_builder, p_grid: ParameterGrid, shared_storage):

        # model (Simulation class) 
        self.sim_class = sim_class
        self.sim_class_signature = compute_class_source_hash_recursive(self.sim_class)
        self.sim_class_file = get_definition_file_of(self.sim_class)
        self.sim_class_file_source = open(self.sim_class_file, 'r').read()
        self.sim_class_file_hash = hash_data(self.sim_class_file_source)

        # mesh builder (function that dumps mesh files)
        self.meshes_dir = os.path.join(shared_storage, "meshes")
        self.mesh_builder = CachedGeometryBuilder(mesh_builder, self.meshes_dir)
        self.mesh_builder_signature = compute_function_source_hash_recursive(mesh_builder)
        self.mesh_builder_file = get_definition_file_of(mesh_builder)
        self.mesh_builder_file_source = open(self.mesh_builder_file, 'r').read()
        self.mesh_builder_file_hash = hash_data(self.mesh_builder_file_source)

        # add data to ParameterGrid
        self.p_grid = p_grid
        self.p_grid.dict.update({
            'SIM_CLASS_NAME': self.sim_class.__name__,
            'SIM_CLASS_SIGNATURE': self.sim_class_signature,
            'SIM_CLASS_FILENAME': self.sim_class_file,
            'SIM_CLASS_FILE_SOURCE': self.sim_class_file_source,
            'SIM_CLASS_FILE_HASH': self.sim_class_file_hash,

            'MESH_BUILDER_SIGNATURE': self.mesh_builder_signature,
            'MESH_BUILDER_FILENAME': self.mesh_builder_file,
            'MESH_BUILDER_FILE_SOURCE': self.mesh_builder_file_source,
            'MESH_BUILDER_FILE_HASH': self.mesh_builder_file_hash,

            'GROUP_SIGNATURE': combine_hashes([self.sim_class_file_hash, self.mesh_builder_file_hash]),
            })

        self.p_grid.instantiate()
        self.unique_mesh_p = self.p_grid.get_unique_by_param_path(['mesh_p'])
    
    def check_mesh_files(self, dir, spec_id, verbose=False):
        """
        Check if all the expected mesh files exist for a given spec_id in a directory.
        """
        msh_path = os.path.join(dir, f'{spec_id}.msh')
        step_path = os.path.join(dir, f'{spec_id}.step')
        ad_path = os.path.join(dir, f'{spec_id}.ad')

        msh_ok = os.path.exists(msh_path)
        step_ok = os.path.exists(step_path)
        ad_ok = os.path.exists(ad_path)

        all_ok = msh_ok and step_ok and ad_ok
        if verbose and not all_ok and MPI.COMM_WORLD.rank == 0:
            print(f"Missing files for {os.path.join(dir, spec_id)}")
            if not msh_ok:
                print(f"    MSH:  {msh_path}")
            if not step_ok:
                print(f"    STEP: {step_path}")
            if not ad_ok:
                print(f"    AD:   {ad_path}")
        return msh_ok, step_ok, ad_ok
    
    def prebuild_meshes(self, parallel=False, error_on_fail=False, verbose=False):
        """
        Generate all meshes (or load from cache) that are needed for the simulations.

        Args:
            parallel: A boolean that indicates if the mesh generation should be parallelized (using mpi4py).
            error_on_fail: A boolean that indicates if the function should raise an error if some of the meshes fail to generate.
            verbose: A boolean that indicates if the function should print information about the success/failure of the mesh generation.
        """

        #generate meshes or load from cache
        dirs, spec_ids, counts = self.mesh_builder.get_multiple(
            self.unique_mesh_p, 
            parallel=parallel, 
            error_on_fail=error_on_fail, 
            return_counts=True,
        )
        n_success, n_failed, n_cached, n_generated = counts

        # check if all the expected mesh files exist
        n_expected = len(self.unique_mesh_p)
        n_msh_ok = 0
        n_step_ok = 0
        n_ad_ok = 0
        all_ok = True
        for dir, spec_id in zip(dirs, spec_ids):
            msh_ok, step_ok, ad_ok = self.check_mesh_files(dir, spec_id, verbose)
            n_msh_ok += int(msh_ok)
            n_step_ok += int(step_ok)
            n_ad_ok += int(ad_ok)
            all_ok *= msh_ok==n_expected and step_ok==n_expected and ad_ok==n_expected

        if not all_ok and verbose and MPI.COMM_WORLD.rank == 0:
            print(f"WARNING!!: Process did not caused errors, but some files that were expected are missing:")
            print('Please look above ↑↑↑ for more information about the missing files.')
            print(f"Success: {n_success}, Failed: {n_failed}, From cache: {n_cached}, Newly generated: {n_generated}")
            print(f"Checking for {n_expected} unique meshes:")
            print(f"  {n_msh_ok} .msh files detected", " - OK" if n_msh_ok == n_expected else " - CHECK FAILED")
            print(f"  {n_step_ok} .step files detected", " - OK" if n_step_ok == n_expected else " - CHECK FAILED")
            print(f"  {n_ad_ok} .ad files detected", " - OK" if n_ad_ok == n_expected else " - CHECK FAILED")

        return dirs, spec_ids
    
    def list_jobs(self):
        return list(self.p_grid.kde_parameters())

    def request_job(self, p_inputs):
        """This method inserts a job into the jobs database."""
        pass

    def run_all_jobs_local(self):
        for p_inputs in self.p_grid.kde_parameters():
            self.run_unsteady(p_inputs)
            break

    def run_steady(self, p_inputs):
        geometry_dir, model_name = self.mesh_builder.get_single(**p_inputs['mesh_p'])
        sim = self.sim_class(geometry_dir=geometry_dir, model_name=model_name)
        sim.solve_steady(**p_inputs['sim_p'])
        return None
    
    def run_unsteady(self, p_inputs):
        geometry_dir, model_name = self.mesh_builder.get_single(**p_inputs['mesh_p'])
        sim = self.sim_class(geometry_dir=geometry_dir, model_name=model_name)
        sim.solve_unsteady(**p_inputs['sim_p'])
        return None
