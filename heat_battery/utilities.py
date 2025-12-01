from mpi4py import MPI
import hashlib
import cloudpickle
import yaml
import json
import os
import time
import datetime
import inspect
from typing import Callable
from pathlib import Path
from typing import Set
import ast
import importlib
import pandas as pd
import numpy as np

def test_package():
    #TODO: make this proper
    return 0

def only_rank_0(f):
    def wrapper(*args, **kwargs):
        if MPI.COMM_WORLD.rank == 0:
            return f(*args, **kwargs)
    return wrapper

def print_rank_0(*args, **kwargs):
    for arg in args:
        if isinstance(arg, object):
            args = (str(arg), ) + args[1:]
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)

def save_data_binary(
        filepath:str, 
        data:any, 
        only_root:bool=True
    ) -> None:
    if not only_root or MPI.COMM_WORLD.rank == 0:
        save_dir, _ = os.path.split(filepath) 
        os.makedirs(save_dir, exist_ok=True)
        with open(filepath, 'wb') as fp:
            cloudpickle.dump(data, fp)
    return None

def load_data_binary(filepath:str) -> any:
    """
    Loads data from a binary file.

    Args:
        filepath (str): Path to the binary file.

    Returns:
        any: Loaded data.
    """
    with open(filepath, 'rb') as fp:
        data = cloudpickle.load(fp)
    return data

def save_data_json(
        filepath:str, 
        data:any, 
        only_root:bool=True
    ) -> None:
    if not only_root or MPI.COMM_WORLD.rank == 0:
        save_dir, _ = os.path.split(filepath) 
        os.makedirs(save_dir, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    return None

def load_data_json(filepath:str) -> any:
    """
    Loads data from a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        any: Loaded data.
    """
    with open(filepath, 'rb') as fp:
        data = json.load(fp)
    return data

def load_yaml(filepath:str) -> dict:
    """
    Loads data from a YAML file.

    Args:
        filepath (str): Path to the YAML file.

    Returns:
        dict: Loaded data.
    """
    with open(filepath, 'rb') as fp:
        data = yaml.safe_load(fp)
    return data

def save_yaml(filepath:str, data:dict) -> None:
    """
    Saves data to a YAML file.

    Args:
        filepath (str): Path to the YAML file.
        data (dict): Data to save.
    """
    with open(filepath, 'wb') as fp:
        yaml.dump(data, fp)
    return None

def get_definition_file_of(_object):
    return inspect.getfile(_object)

def hash_data(data:any) -> str:
    """
    Hashes data using SHA256.

    Args:
        data (any): Data to hash.

    Returns:
        str: Hashed data.
    """
    m = hashlib.sha256()
    if isinstance(data, bytes):
        m.update(data)
    else:
        d_str = json.dumps(data)
        m.update(d_str.encode('UTF-8'))
    return m.hexdigest()

def combine_hashes(hashes:list) -> str:
    """
    Combines multiple hashes into a single hash by hashing the hashes.

    Args:
        hashes (list): List of hashes to combine.

    Returns:
        str: Combined hash.
    """
    m = hashlib.sha256()
    for h in hashes:
        m.update(h.encode('UTF-8'))
    return m.hexdigest()

def compute_function_source_hash(func:Callable) -> str:
    """
    Computes the hash of a function's source code.

    Args:
        func (function): Function to hash.

    Returns:
        str: Hashed function source code.
    """
    return hash_data(inspect.getsource(func))

def get_function_combined_source(func, only_from_modules=None):
    """
    Gets the combined source code of a function and all functions it calls.

    Args:
        func (function): Function to get the combined source code of.
        only_from_modules (list, optional): List of modules to include in the
        combined source code. Default value None includes all modules.

    Returns:
        str: Combined source code of the function and all functions it calls.
    """
    top_level_source = inspect.getsource(func)

    #FIXME: this recursive search is not working as expected
    #       changes in source code of called functions are not reflected
    called_funcs = [obj for name, obj in inspect.getmembers(func.__globals__)
                    if inspect.isfunction(obj) and obj.__name__ in top_level_source
                    and (only_from_modules is None or inspect.getmodule(obj) in only_from_modules)]
    
    inner_combided_sources = [get_function_combined_source(f, only_from_modules) for f in called_funcs]
    return top_level_source + ''.join(inner_combided_sources)

def compute_function_source_hash_recursive(func, only_from_modules=None):
    """
    Computes the hash of a function's source code and all functions it calls.

    Args:
        func (function): Function to compute the hash of.
        only_from_modules (list, optional): List of modules to include in the
        combined source code. Default value None includes all modules.

    Returns:
        str: Hashed function source code and all functions it calls.
    """
    combined_source = get_function_combined_source(func, only_from_modules)
    return hash_data(combined_source)

def compute_class_source_hash(cls):
    """
    Computes the hash of a class's source code.

    Args:
        cls (class): Class to compute the hash of.

    Returns:
        str: Hashed class source code.
    """
    return hash_data(inspect.getsource(cls))

def compute_class_source_hash_recursive(cls):
    """
    Computes the hash of a class's source code and all classes it inherits from.
    
    Args:
        cls (class): Class to compute the hash of.

    Returns:
        str: Hashed class source code and all classes it inherits from.
    """
    class_source = inspect.getsource(cls)
    
    parent_sources = []
    for parent in cls.__bases__:
        parent_sources.append(inspect.getsource(parent))
    
    combined_source = class_source + ''.join(parent_sources)
    
    # Compute hash of the combined source
    return hash_data(combined_source)

class ProgressBar():
    def __init__(self, desc="", update_cb=None, n=20):
        self.desc = desc
        self.update_cb = update_cb
        self.percentage = 0.0
        self.prev_percentage = 0.0
        self.n = n
        self.fill_char = "█"
        self.unfill_char = "░"
        self.start_t = time.time()
        self.last_t = self.start_t
        self.update()

    def update(self):
        self.percentage = self.update_cb()
        t = time.time()

        self.l_bar = f"{self.desc}: {self.percentage:3.3f}%"

        filled_n = int(self.percentage/100*self.n)
        unfiled_n = self.n - filled_n
        self.bar = filled_n*self.fill_char+unfiled_n*self.unfill_char

        elapsed = round(t - self.start_t)
        remain = 0 if self.percentage < 1e-9 else elapsed/(self.percentage)*(100-self.percentage)
        if self.percentage < 100.0:
            e_str = str(datetime.timedelta(seconds=elapsed))
            r_str = str(datetime.timedelta(seconds=remain))
            self.r_bar = f"{e_str}>{r_str}"
        else:
            e_str = str(datetime.timedelta(seconds=elapsed))
            self.r_bar = f"Finished in {e_str}"

        self.print_bar()

        self.last_t = t
        self.prev_percentage = self.percentage

    def finish(self):
        pass

    def print_bar(self):
        if MPI.COMM_WORLD.rank == 0:
            if self.percentage < 100.0:
                print(f"\033[1K\r{self.l_bar}|{self.bar}|{self.r_bar}", end="", flush=True)
            else:
                print(f"{self.l_bar}|{self.bar}|{self.r_bar}", end="\n", flush=True)

    def print_message(self, str):
        if MPI.COMM_WORLD.rank == 0:
            print(f"\033[1K\r{str}", flush=True)
            self.print_bar()

def resolve_dependencies_of_file(
        file_path: str, 
        analyzed_files: Set[str] = None, 
        constraint_path: str = None,
    ) -> Set[str]:
    """
    Recursively resolves the dependencies of a single python source file.

    Args:
        file_path (str): Path to the python source file to resolve the
        dependencies of. analyzed_files (set[str], optional): Set of file paths
        that have already been analyzed. Used to prevent infinite loops.
        constraint_path (str, optional): Path to constrain the search for
        dependencies.

    Returns:
        set[str]: Set of python source file paths that are imported or imported
        from directly by the input file or by the modules it depends on.
    """
    
    if analyzed_files is None:
        analyzed_files = set()
    
    file_path = str(Path(file_path).resolve())
    if file_path in analyzed_files:
        return set()
    
    analyzed_files.add(file_path)
    dependencies = set()
    
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
            
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.level > 0:
                    # Handle relative imports
                    parent = Path(file_path)
                    for _ in range(node.level):
                        parent = parent.parent
                    module_path = parent / (node.module if node.module else '')
                else:
                    # Handle absolute imports
                    module_name = node.names[0].name.split('.')[0]
                    spec = importlib.util.find_spec(module_name)
                    if spec is None or spec.origin is None:
                        continue
                    module_path = Path(spec.origin)
                    if not str(module_path).startswith(constraint_path):
                        continue
                
                if module_path.with_suffix('.py').exists():
                    # Handle single file modules 
                    dep_path = str(module_path.with_suffix('.py').resolve())
                    dependencies.add(dep_path)
                    dependencies.update(
                        resolve_dependencies_of_file(
                            dep_path, 
                            analyzed_files, 
                            constraint_path=constraint_path
                        )
                    )
                elif module_path.exists():
                    # Handle module organised in a folder
                    files = [
                        str(module_path / f) 
                        for f in os.listdir(module_path) 
                        if f.endswith('.py')
                    ]
                    for dep_path in files:
                        dependencies.add(dep_path)
                        dependencies.update(
                            resolve_dependencies_of_file(
                                dep_path, 
                                analyzed_files, 
                                constraint_path=constraint_path
                            )
                        )
                    
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        
    return dependencies

def resolve_files_dependencies(
        file_paths:list[str],
        constraint_path:str = None,
    ) -> list[str]:
    """
    Resolves the dependencies of a list of python source files.

    Args:
        file_paths (list[str]): List of file paths to resolve the dependencies
        of. constraint_path (str, optional): Path to constrain the search for
        dependencies.

    Returns:
        list[str]: List of python source file paths that are imported directly
        by the input files or by the modules they depend on.
    """
    dependencies = set()
    for file_path in file_paths:
        dependencies.update(
            resolve_dependencies_of_file(
                file_path, 
                constraint_path=constraint_path
            )
        )
    return list(dependencies)
