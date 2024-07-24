import csv
import pandas as pd
from mpi4py import MPI
from dolfinx import geometry, io
import numpy as np
import pandas as pd
import os

class FunctionSampler:
    def __init__(self, p_coords, domain):
        self.domain = domain
        points = np.array(p_coords)
        bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points)
        self.cells = []
        self.points_on_proc = []
        self.idx_on_proc = []
        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                self.points_on_proc.append(point)
                self.cells.append(colliding_cells.links(i)[0])
                self.idx_on_proc.append(i)
        self.points_on_proc = np.array(self.points_on_proc, dtype=np.float64)
        self.idx_on_proc = np.array(self.idx_on_proc, dtype=int)

    def eval(self, f):
        fv = f.eval(self.points_on_proc, self.cells)
        fv = self.domain.comm.allgather(fv)
        idx = self.domain.comm.allgather(self.idx_on_proc)
        fv = np.vstack(fv)
        idx = np.concatenate(idx)
        sorted_idx_arg = np.argsort(idx) # sort ascending (might contain duplicates)
        fv = fv[sorted_idx_arg]    # reorder values to original order of the p_coords
        idx = idx[sorted_idx_arg]  # indexes to original order (duplicates still in)
        _, u_idx = np.unique(idx, return_index=True) # create idexing for unique values
        fv = fv[u_idx] # remove duplicates from values
        return fv.flatten()

class Probe_writer:
    def __init__(self, result_dir, default_file_name='probes.csv', flush=True):
        self.names = []
        self.pretty_names = []
        self.formats = []
        self.probes = []
        self.values = []
        self.units = []
        self.name_map = {}
        self.result_dir = result_dir
        self.file_name = default_file_name
        self.is_file_initialized = False
        self.is_memory_initialized = False
        self.flush = flush
        self.printer = print

    def set_printer(self, printer):
        self.printer = printer

    def set_result_file_name(self, file_path):
        assert not self.is_file_initialized, "Cannot change file path after initialisation."
        self.file_name = file_path

    def reset_printer(self):
        self.printer = print
    
    def print_r0(self, str):
        if MPI.COMM_WORLD.rank == 0:
            self.printer(str)

    def register_probe(self, name="unspecified", unit=" ", pretty_name=None, format=None):
        def decorator(f):
            self.name_map[name] = len(self.names)
            self.pretty_names.append(pretty_name)
            self.formats.append(format)
            self.names.append(name)
            self.units.append(unit)
            self.probes.append(f)
            self.values.append(0.0)
        return decorator
    
    def evaluate_probes(self):
        # evaluate probes with MPI
        values = []
        for i, probe in enumerate(self.probes):
            try:
                value = probe()
            except Exception as e:
                value = 0
            self.values[i] = value

    def chain_names(self):
        names_chain = []
        for i, value in enumerate(self.values):
            if hasattr(value, '__iter__'):
                for j in range(len(value)):
                    names_chain.append(self.names[i] + f'[{j}]')
            else:
                names_chain.append(self.names[i])
        return names_chain
    
    def chain_values(self):
        values_chain = []
        for value in self.values:
            if hasattr(value, '__iter__'):
                for item in value:
                    values_chain.append(item)
            else:
                 values_chain.append(value)
        return values_chain

    def write_header(self, header_names):
        if MPI.COMM_WORLD.rank == 0:
            if not os.path.exists(self.result_dir):
                self.print_r0(f"Making new result directory: {self.result_dir}")
                os.makedirs(self.result_dir)
            result_file_path = os.path.join(self.result_dir, self.file_name)
            self.file = open(result_file_path, 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow(header_names)

    def initialize_file(self):
        if MPI.COMM_WORLD.rank == 0:
            header_names = self.chain_names()
            self.write_header(header_names)
            self.print_r0(f"New probe file opened: {self.file}")
        self.is_file_initialized = True

    def initialize_memory(self):
        if MPI.COMM_WORLD.rank == 0:
            header_names = self.chain_names()
            self.df = pd.DataFrame(columns=header_names)
        self.is_memory_initialized = True

    def write_probes_to_file(self):
        # initialise header in serial before first entry
        if not self.is_file_initialized:
            self.initialize_file()

        # append row to file
        if MPI.COMM_WORLD.rank == 0:
            values_chain = self.chain_values()
            self.writer.writerow(values_chain)
            if self.flush:
                self.file.flush()

    def write_probes_to_memory(self):
        # initialise header in serial before first entry
        if not self.is_memory_initialized:
            self.initialize_memory()

        # append row to dataframe
        if MPI.COMM_WORLD.rank == 0:
            values_chain = self.chain_values()
            self.df.loc[len(self.df)] = values_chain

    def get_value(self, name, use_pretty_names=False):
        '''Get current evaluated value of a probe'''
        i = self.name_map[name]
        v = self.values[i]
        if use_pretty_names:
            pn = self.pretty_names[i]
            assert pn is not None, "Pretty names are not set for this probe."
            d = {pn[j]:v[j] for j in range(len(v))}
            return d
        else:
            return v
    
    def get_values(self, names):
        if isinstance(names, list):
            return [self.get_value(name) for name in names]
        else:
            return self.get_values(names)

    def pretty_string(self):
        texts = []
        for i in range(len(self.probes)):
            name = self.pretty_names[i] if self.pretty_names[i] is not None else self.names[i]
            format = self.formats[i] if self.formats[i] is not None else '.4f'
            if hasattr(self.values[i], '__iter__'):
                value_string = [f"{item:{format}}" for item in self.values[i]]
            else:
                value_string = f"{self.values[i]:{format}}"
            texts.append(f'{name}={value_string}[{self.units[i]}]')
        return ",".join(texts)
    
    def pretty_print(self):
        self.print_r0(self.pretty_string())

    def close(self):
        if MPI.COMM_WORLD.rank == 0:
            if hasattr(self, 'file'):
                self.file.close()
                self.print_r0(f"Probe file closed: {self.file}")
        self.is_file_initialized = False
        self.is_memory_initialized = False

    def head(self):
        return self.df.head()


    def __str__(self):
        return str(self.df)

