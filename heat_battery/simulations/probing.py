import csv
import pandas as pd
from mpi4py import MPI
from dolfinx import geometry
import numpy as np
import pandas as pd
import os

from psycopg2 import sql
from ..database.postgresql_connection import get_single_db_connection, safe_query
from ..config import get_config_item

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
        _, u_idx = np.unique(idx, return_index=True) # create indexing for unique values
        fv = fv[u_idx] # remove duplicates from values
        return fv.flatten()

class UnitPrettyPrinter:
    up_alt_names_short = ['k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    up_alt_names_full = ['kilo', 'mega', 'giga', 'tera', 'peta', 'exa', 'zetta', 'yotta']
    down_alt_names_short = ['m', 'µ', 'n', 'p', 'f', 'a', 'z', 'y']
    down_alt_names_full = ['milli', 'micro', 'nano', 'pico', 'femto', 'atto', 'zepto', 'yocto']
    def __init__(self, base_name, format='.2f'):
        self.base_name = base_name
        self.format = format

    def to_base_string(self, value):
        return f"{value:{self.format}}[{self.base_name}]"

    def to_alt_string(self, value):
        base10 = np.floor(np.log10(abs(value)))
        thousands = int(base10/3)
        if thousands < 0:
            alt_name = UnitPrettyPrinter.down_alt_names_short[abs(thousands)]
        elif thousands > 0:
            alt_name = UnitPrettyPrinter.up_alt_names_short[abs(thousands)]
        else:
            alt_name = ""
        return f"{value*10**(-thousands):{self.format}}[{alt_name}{self.base_name}]"

class Probe_writer:
    def __init__(self, flush=True):
        self.names = []
        self.pretty_names = []
        self.formats = []
        self.probes = []
        self.values = []
        self.descriptions = []
        self.units = []
        self.name_map = {}
        self.result_dir = None
        self.file_name = None
        self.result_database = None
        self.table_name = None
        self.project_name = None
        self.is_file_initialized = False
        self.is_database_table_initialized = False
        self.is_memory_initialized = False
        self.flush = flush
        self.printer = print

    def set_printer(self, printer):
        self.printer = printer

    def set_result_file(
        self, 
        result_dir: os.PathLike | str, 
        file_name: os.PathLike | str
        ) -> None:
        assert not self.is_file_initialized, "Cannot change result file after initialisation."
        if result_dir is not None and file_name is not None:
            self.result_dir = result_dir
            self.file_name = file_name
        else:
            raise ValueError("One of the required parameters is set to None.")

    def set_result_database_table(
        self, 
        result_database: os.PathLike | str, 
        project_name: os.PathLike | str, 
        table_name: os.PathLike | str
        ) -> None:
        assert not self.is_database_table_initialized, "Cannot change database output after initialisation."
        if result_database is not None and project_name is not None and table_name is not None:
            self.result_database = result_database
            self.project_name = project_name
            self.table_name = table_name

    def set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def reset_printer(self):
        self.printer = print
    
    def print_r0(self, str):
        if MPI.COMM_WORLD.rank == 0:
            self.printer(str)

    def register_probe(self, name="unspecified", unit=" ", pretty_name=None, format=None, description=None):
        assert self.name_map.get(name) is None, f"Probe named {name} already registered. Choose different name."
        def decorator(f):
            self.name_map[name] = len(self.names)
            self.pretty_names.append(pretty_name)
            self.formats.append(format)
            self.descriptions.append(description)
            self.names.append(name)
            self.units.append(unit)
            self.probes.append(f)
            self.values.append(0.0)
        return decorator
    
    def register_field_probe(self, name, unit, description=None):
        def decorator(f):
            self.name_map[name] = len(self.names)
            self.pretty_names.append(name)
            self.formats.append(None)
            self.descriptions.append(description)
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
        self.chained_values = self.chain_values()

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

    def initialize_result_file(self):
        if MPI.COMM_WORLD.rank == 0:
            assert self.result_dir is not None, "Result directory must be set before initialising a file."
            assert self.file_name is not None, "File name must be set before initialising a file."
            if not os.path.exists(self.result_dir):
                self.print_r0(f"Making new result directory: {self.result_dir}")
                os.makedirs(self.result_dir)
            result_file_path = os.path.join(self.result_dir, self.file_name)
            if os.path.exists(result_file_path):
                self.print_r0(f"Probe file already exists: {result_file_path}. Overwriting.")
            self.file = open(result_file_path, 'w', newline='')
            self.print_r0(f"New probe file opened: {self.file}")
            self.writer = csv.writer(self.file)
            header_names = self.chain_names()
            self.writer.writerow(header_names)
        self.is_file_initialized = True

    @safe_query
    def initialize_database_table(self):

        if MPI.COMM_WORLD.rank == 0:
            assert self.result_database is not None, "Database must be set before initialising a database table."
            assert self.project_name is not None, "Project name must be set before initialising a database table."
            assert self.table_name is not None, "Table name must be set before initialising a database table."

            column_specs = [
                sql.SQL("{} REAL").format(sql.Identifier(name))
                for name in self.chain_names()
            ]

            query = sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ({})"
            )
            query = query.format(
                sql.Identifier(self.project_name, self.table_name),
                sql.SQL(', ').join(column_specs)
            )
            self.print_r0(f"Creating new result table: {self.table_name} in project: {self.project_name} in database: {self.result_database}.")
            self.conn = get_single_db_connection(self.result_database)
            cur = self.conn.cursor()
            cur.execute(query)
            self.conn.commit()

            self.insert_query = sql.SQL(
                "INSERT INTO {} ({}) VALUES ({})"
            )
            self.insert_query = self.insert_query.format(
                sql.Identifier(self.project_name, self.table_name),
                sql.SQL(', ').join([sql.Identifier(name) for name in self.chain_names()]),
                sql.SQL(', ').join([sql.SQL('%s')]*len(self.chain_names()))
            )
        self.is_database_table_initialized = True

    def initialize_memory(self):
        if MPI.COMM_WORLD.rank == 0:
            header_names = self.chain_names()
            self.df = pd.DataFrame(columns=header_names)
        self.is_memory_initialized = True
    
    def write_probes_to_file(self):
        # automatically initialise file if not already done
        if not self.is_file_initialized:
            self.initialize_result_file()

        # append row to file
        if MPI.COMM_WORLD.rank == 0:
            self.writer.writerow(self.chained_values)
            if self.flush:
                self.file.flush()

    @safe_query
    def write_probes_row_to_database(self):
        if MPI.COMM_WORLD.rank == 0:
            cur = self.conn.cursor()
            cur.execute(self.insert_query, self.chained_values)
            self.conn.commit()

    def write_probes_to_database_table(self):
        # automatically initialise table if not already done
        if not self.is_database_table_initialized:
            self.initialize_database_table()

        # append row to database table
        self.write_probes_row_to_database()

    def write_probes_to_memory(self):
        # automatically initialise memory if not already done
        if not self.is_memory_initialized:
            self.initialize_memory()

        # append row to dataframe
        if MPI.COMM_WORLD.rank == 0:
            self.df.loc[len(self.df)] = self.chained_values

    def write_all_set_probes_outputs(self):
        if (self.result_dir is not None 
                and self.file_name is not None):
                self.write_probes_to_file()
        if (self.result_database is not None 
                and self.project_name is not None 
                and self.table_name is not None):
            self.write_probes_to_database_table()
        self.write_probes_to_memory()
        for callback in self.callbacks:
            callback(self)

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

    def close_result_file(self):
        if MPI.COMM_WORLD.rank == 0:
            if hasattr(self, 'file'):
                self.file.close()
                self.print_r0(f"Probe file closed: {self.file}")
        self.is_file_initialized = False

    def close_database_connection(self):
        if hasattr(self, 'conn'):
            self.conn.close()
            self.print_r0(f"Database connection closed.")
        self.is_database_table_initialized = False

    def close_all(self):
        self.close_result_file()
        self.close_database_connection()

    def head(self):
        return self.df.head()

    def __str__(self):
        return str(self.df)


