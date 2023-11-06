import csv
import pandas as pd
from mpi4py import MPI
from dolfinx import geometry
import numpy as np
from termcolor import colored

def probe_function(p_coords, domain, f):
    points = np.array(p_coords)
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    cells = []
    points_on_proc = []
    idx = []
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
            idx.append(i)
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    idx = np.array(idx, dtype=int)
    fv = f.eval(points_on_proc, cells)
    fv = domain.comm.allgather(fv)
    idx = domain.comm.allgather(idx)
    fv = np.vstack(fv)
    idx = np.concatenate(idx)
    fv = fv[np.argsort(idx)]    # reorder values to original order of the p_coords
    idx = idx[np.argsort(idx)]  # indexes to original order (there might be duplicates)
    _, u_idx = np.unique(idx, return_index=True) # remove duplicates from indexing
    fv = fv[u_idx] # remove duplicates from temperatures
    return fv.flatten()

class Probe_writer:
    def __init__(self, file_path, flush=True):
        self.names = []
        self.pretty_names = []
        self.formats = []
        self.probes = []
        self.values = []
        self.units = []
        self.name_map = {}
        self.file_path = file_path
        self.initialized = False
        self.flush = flush

    def register_probe(self, name="unspecified", unit=" ", pretty_name=None, format=None):
        def decorator(f):
            self.name_map[name] = len(self.names)
            self.pretty_names.append(pretty_name)
            self.formats.append(format)
            self.names.append(name)
            self.units.append(unit)
            self.probes.append(f)
        return decorator
    
    def evaluate_probes(self):
        # evaluate probes with MPI
        values = []
        for i, probe in enumerate(self.probes):
            try:
                value = probe()
            except Exception as e:
                value = 0
            values.append(value)
        self.values = values

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
            self.file = open(self.file_path, 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow(header_names)

    def write_probes(self):
        # initialise header in serial before first entry
        if not self.initialized:
            if MPI.COMM_WORLD.rank == 0:
                header_names = self.chain_names()
                self.write_header(header_names)
            self.initialized = True

        # append row to file
        if MPI.COMM_WORLD.rank == 0:
            values_chain = self.chain_values()
            self.writer.writerow(values_chain)
            if self.flush:
                self.file.flush()

    def get_value(self, name):
        '''Get current lat evaluated value of a probe'''
        i = self.name_map[name]
        return self.values[i]
    
    def pretty_string(self):
        texts = []
        for i in range(len(self.probes)):
            name = self.pretty_names[i] if self.pretty_names[i] is not None else self.names[i]
            format = self.formats[i] if self.formats[i] is not None else '.3f'
            if hasattr(self.values[i], '__iter__'):
                value_string = [f"{item:{format}}" for item in self.values[i]]
            else:
                value_string = f"{self.values[i]:{format}}"
            texts.append(f'{name}={value_string}[{self.units[i]}]')
        return ",".join(texts)
    
    def print(self):
        if MPI.COMM_WORLD.rank == 0:
            print(self.pretty_string())

    def close(self):
        if MPI.COMM_WORLD.rank == 0:
            if hasattr(self, 'file'):
                self.file.close()
