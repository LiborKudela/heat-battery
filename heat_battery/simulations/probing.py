import csv
import pandas as pd
from mpi4py import MPI
from dolfinx import geometry
import numpy as np
import pandas as pd
import os
import shutil
import tempfile

from psycopg2 import sql
from ..database.postgresql_connection import get_single_db_connection, debounced_query, DBConnection
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

class ProbeWriteDestination:
    def __init__(self):
        pass

    def initialize(self):
        pass

    def write_row(self, values):
        pass

    def close(self):
        pass

    def set_printer(self, printer):
        self.printer = printer

    def reset_printer(self):
        self.printer = print

class MemoryProbeWriteDestination(ProbeWriteDestination):
    def __init__(self):
        self.df = pd.DataFrame()
        self.is_initialized = False
        self.printer = print

    @staticmethod
    def from_dict(d):
        return MemoryProbeWriteDestination()

    def initialize(self, header_names):
        if self.is_initialized:
            raise ValueError("MemoryProbeWriteDestination already initialized.")

        if MPI.COMM_WORLD.rank == 0:
            self.df = pd.DataFrame(columns=header_names)
        self.is_initialized = True

    def write_row(self, values):
        self.df.loc[len(self.df)] = values

    def close(self):
        pass

class CSVFileProbeWriteDestination(ProbeWriteDestination):
    def __init__(
            self,
            result_dir: os.PathLike | str, 
            file_name: os.PathLike | str, 
            flush: bool = True
        ):
        self.result_dir = result_dir
        self.file_name = file_name
        self.flush = flush
        self.is_initialized = False
        self.printer = print

    @staticmethod
    def from_dict(d: dict):
        REQUIRED_KEYS = ['result_dir', 'file_name']
        for key in REQUIRED_KEYS:
            if key not in d:
                raise ValueError(f"Missing required key: {key}.")
        return CSVFileProbeWriteDestination(
            result_dir=d.get('result_dir'),
            file_name=d.get('file_name'),
            flush=d.get('flush'),
        )

    def initialize(
            self, 
            header_names: list[str],
            mode: str = 'w',
        ):
        assert mode in ['w', 'a'], "Mode must be 'w' or 'a'."
        if self.is_initialized:
            raise ValueError("CSVFileProbeWriteDestination already initialized.")
        if MPI.COMM_WORLD.rank == 0:
            if not os.path.exists(self.result_dir):
                self.printer(f"Making new result directory: {self.result_dir}")
                os.makedirs(self.result_dir)
            result_file_path = os.path.join(self.result_dir, self.file_name)
            if os.path.exists(result_file_path):
                self.printer(f"Probe file already exists: {result_file_path}. Overwriting.")
            self.file = open(result_file_path, mode, newline='')
            self.printer(f"New probe file opened: {self.file}")
            self.writer = csv.writer(self.file)
            if mode == 'w':
                self.writer.writerow(header_names)
        self.is_initialized = True

    def write_row(
            self, 
            values: list[float]
        ):
        if MPI.COMM_WORLD.rank == 0:
            self.writer.writerow(values)
            if self.flush:
                self.file.flush()

    def close(self):
        if MPI.COMM_WORLD.rank == 0:
            self.file.close()

class DatabaseProbeWriteDestination(ProbeWriteDestination):
    def __init__(
            self,
            result_database: str,
            project_name: str,
            table_name: str
        ):
        self.result_database = result_database
        self.project_name = project_name
        self.table_name = table_name
        self.printer = print
        self.is_initialized = False

    @staticmethod
    def from_dict(d: dict):
        REQUIRED_KEYS = ['result_database', 'project_name', 'table_name']
        for key in REQUIRED_KEYS:
            if key not in d:
                raise ValueError(f"Missing required key: {key}.")
        
        return DatabaseProbeWriteDestination(
            result_database=d.get('result_database'),
            project_name=d.get('project_name'),
            table_name=d.get('table_name'),
        )

    @debounced_query
    def _initialize_query(
        self, 
        header_names: list[str],
        mode: str = 'w',
    ):
        assert mode in ['w', 'a'], "Mode must be 'w' or 'a'."
        if MPI.COMM_WORLD.rank == 0:
            self.conn = get_single_db_connection(self.result_database)
            prefix = get_config_item(['database', 'postgres', 'bin_files_folder_prefix'])
            self.res_prefix = os.path.join(prefix, self.project_name, 'results')
            self.binary_file_name = os.path.join(self.res_prefix, f'{self.table_name}.hbres')

            # check if file exists
            query = sql.SQL("SELECT {}.check_file_exists(%s)")
            query = query.format(sql.Identifier(self.project_name))
            cur = self.conn.cursor()
            cur.execute(query, (self.binary_file_name,))
            res = cur.fetchone()[0]
            if res:
                self.printer(f"Probe binary file {self.binary_file_name} already exists in database. Overwriting.")
            else:
                self.printer(f"Creating new probe binary file in database: {self.binary_file_name}.")

            # binary file header data
            col_names = header_names
            col_names_bytes = bytes('\n'.join(col_names), 'utf-8')
            n_cols_bytes = np.int64(len(col_names)).tobytes()
            n_cols_names_bytes = np.int64(len(col_names_bytes)).tobytes()
            header_bytes = n_cols_bytes + n_cols_names_bytes + col_names_bytes

            if mode == 'w':
                query = sql.SQL("SELECT {}.write_bytes_to_file(%s, %s)")
                query = query.format(sql.Identifier(self.project_name))
                cur = self.conn.cursor()
                cur.execute(query, (self.binary_file_name, header_bytes))  
            
                # write colums to job table
                query = sql.SQL("UPDATE {} SET {} = %s")
                query = query.format(
                    sql.Identifier(self.project_name, 'jobs'),
                    sql.Identifier('probe_columns'),
                )
                cur = self.conn.cursor()
                cur.execute(query, (header_names,))
                self.conn.commit()
            elif mode == 'a':
                # check that colums agree
                query = sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s")
                query = query.format(sql.Identifier(self.project_name, 'jobs'))
                cur = self.conn.cursor()
                cur.execute(query, (self.binary_file_name,))
                res = cur.fetchall()
                if res != header_names:
                    raise ValueError(
                        "Column names of ProbeWriter do not agree with existing "
                        "file in Database. \n"
                        f"Columns in database: {len(res)} columns -> {res}\n"
                        f"Columns in ProbeWriter: {len(header_names)} columns -> {header_names}\n"
                        "Aborting since this is critical error."
                    )

            self.insert_query = sql.SQL(f"SELECT {self.project_name}.append_bytes_to_file('{self.binary_file_name}', %s)")

    def initialize(
        self, 
        header_names: list[str],
        mode: str = 'w',
    ):
        if self.is_initialized:
            raise ValueError("DatabaseProbeWriteDestination already initialized.")
        self._initialize_query(header_names)
        self.is_initialized = True

    @debounced_query
    def _write_row_query(
        self, 
        values: list[float]
    ):
        if MPI.COMM_WORLD.rank == 0:
            if self.conn.closed > 0: # check if connection is closed
                self.conn = get_single_db_connection(self.result_database)
            cur = self.conn.cursor()
            bytes_row = np.array(values, dtype=np.float64).tobytes()
            cur.execute(self.insert_query, (bytes_row,))
            self.conn.commit()

    def write_row(
        self, 
        values: list[float]
    ):
        self._write_row_query(values)

    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()
            self.printer(f"Database connection closed.")
        self.is_initialized = False

def parse_probe_destinations(d):
    assert isinstance(d, dict), "Probe destinations must be a dictionary."
    assert 'type' in d, "Probe destinations must have a 'type' key."
    ALOWED_TYPES = ['memory', 'csv', 'database']

    assert d['type'] in ALOWED_TYPES, f"Unknown probe destination type: {d['type']}. Allowed types: {ALOWED_TYPES}."
    if d['type'] == 'memory':
        return MemoryProbeWriteDestination.from_dict(d)
    elif d['type'] == 'csv':
        return CSVFileProbeWriteDestination.from_dict(d)
    elif d['type'] == 'database':
        return DatabaseProbeWriteDestination.from_dict(d)
    
class CheckpointDestination:
    def __init__(self, path):
        self.path = path

    def get_temp_dir(self):
        return self.path

    def check_unpacked_checkpoint(self):
        if MPI.COMM_WORLD.rank == 0:
            if os.path.isdir(self.path):
                # check the ADIOS2 stuff
                if os.path.isdir(os.path.join(self.path, 'function')):
                    for item in ['data.0', 'md.0', 'md.idx', 'profiling.json']:
                        item_path = os.path.join(self.path, 'function', item)
                        if not os.path.isfile(item_path):
                            return False
                        
                # check the data of terms and local
                if not os.path.isfile(os.path.join(self.path, 'data.pickle')):
                    return False
                else:
                    return True # valid checkpoint
            else:
                return False
            
    def check_destination(self):
        return self.check_unpacked_checkpoint()

    def archive_checkpoint(self):
        pass

    def unpack_checkpoint(self):
        pass

class ArchiveCheckpointDestination(CheckpointDestination):
    def __init__(self, archive_path, temp_dir, format='zip'):
        self.archive_path = archive_path
        self.temp_dir = temp_dir
        self.format = format

    def get_temp_dir(self):
        return self.temp_dir

    def check_destination(self):
        return os.path.exists(self.archive_path)

    def archive_checkpoint(self):
        self.check_unpacked_checkpoint()
        if MPI.COMM_WORLD.rank == 0:
            shutil.make_archive(
                base_name=self.archive_path,
                format=self.format,
                root_dir=self.temp_dir
            )

    def unpack_checkpoint(self):
        if MPI.COMM_WORLD.rank == 0:
            shutil.unpack_archive(
                filename=self.archive_path,
                extract_dir=self.temp_dir,
                format=self.format
            )
        self.check_unpacked_checkpoint()

    def check_destination(self):
        return os.path.isfile(self.archive_path)

# class DatabaseCheckpointDestination(CheckpointDestination):
#     def __init__(self, db_name, project_name, table_name, format='zip'):
#         self.db_name = db_name
#         self.project_name = project_name
#         self.table_name = table_name
#         self.format = format
#         self.temp_dir = os.path.join(get_config_item(['database', 'postgres', 'res_folder_prefix']), 'checkpoint')

#     def check_destination(self):
#         return False 

#     @debounced_query
#     def _archive_checkpoint_query(
#             self, 
#             checkpoint_dir: str
#         ):
#         self.check_unpacked_checkpoint()
#         if MPI.COMM_WORLD.rank == 0:   
#             shutil.make_archive(
#                 base_name=os.path.join(self.temp_dir, self.table_name),
#                 format=self.format,
#                 root_dir=checkpoint_dir
#             )
#             with DBConnection() as conn:
#                 query = sql.SQL("UPDATE {} SET {} = %s")

#     def unpack_checkpoint(self, checkpoint_dir):
#         if MPI.COMM_WORLD.rank == 0:
#             shutil.unpack_archive(
#                 filename=self.path,
#                 extract_dir=checkpoint_dir,
#                 format=self.format
#             )
        
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
        self.is_initialized = False
        self.destinations = []
        self.printer = print

    def set_printer(self, printer):
        self.printer = printer
        if self.is_initialized:
            printer(
                "WARNING: Changing ProbeWriter printer after initialization. This happens when you call "
                "set_printer() after calling initialize() method.")
        for destination in self.destinations:
            destination.set_printer(self.printer)

    def add_destination(self, destination):
        if self.is_initialized:
            raise ValueError(
                "Cannot add destination after initialization. This happens when "
                "you call add_destination() after calling initialize() method.")
        if isinstance(destination, dict):
            self.destinations.append(parse_probe_destinations(destination))
        elif isinstance(destination, ProbeWriteDestination):
            self.destinations.append(destination)
        else:
            raise ValueError(f"Unknown probe destination type: {type(destination)}.")

    def initialize(self):
        if self.is_initialized:
            raise ValueError(
                "ProbeWriter already initialized. This happens when you call "
                "initialize() more than once before properly closing the "
                "ProbeWriter.")
        self.evaluate_probes() #to get sizes of the individual values
        for destination in self.destinations:
            destination.initialize(self.chain_names())
        self.is_initialized = True

    def set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def reset_printer(self):
        self.printer = print
        for destination in self.destinations:
            destination.reset_printer()
    
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

    def write_all_set_probes_outputs(self):
        if not self.is_initialized:
            raise ValueError(
                "ProbeWriter not initialized. This happens when you call "
                "write_all_set_probes_outputs() before calling initialize() method.")
        for destination in self.destinations:
            destination.write_row(self.chained_values) 
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

    def close(self):
        if not self.is_initialized:
            raise ValueError(
                "ProbeWriter not initialized. This happens when you call "
                "close() before calling initialize() method.")
        for destination in self.destinations:
            destination.close()
        self.is_initialized = False


