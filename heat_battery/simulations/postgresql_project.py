import pandas as pd
from mpi4py import MPI
import datetime
import json
import os
import tarfile
from ..utilities import (
    hash_data,
    only_rank_0,
    print_rank_0,
    load_data_json,
)
from .jobs import Job
import numpy as np
import time
import io

import psycopg2
import psycopg2.sql as sql
from psycopg2.extras import execute_values
from ..database.postgresql_connection import (
    DBConnection, 
    get_single_db_connection, 
    debounced_query
)
from ..config import get_config_item

MAX_WAIT_BEFORE_INTERRUPTED = 60 # seconds

@only_rank_0
def _create_database_query(if_exists:str='skip'):
    conn = get_single_db_connection('postgres')
    conn.autocommit = True
    db_name = get_config_item(['database', 'postgres', 'db_name'])

    query = sql.SQL(
            "SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = %s)"
        )
    cur = conn.cursor()
    cur.execute(query, (db_name,))
    res = cur.fetchone()[0]
    if res:
        if if_exists == 'skip': 
            print(
                f"Database '{db_name}' already exists, "
                "skipping creation"
            )
            return None # <- stop here
        elif if_exists == 'fail':
            raise ValueError(
                f"Database '{db_name}' already exists, "
                "cannot proceed further!"
            )
        else:
            raise ValueError(
                "Unknown value of if_exists. Expected 'skip' or 'fail', "
                f"got '{if_exists}'."
            )

    print(f"Database '{db_name}' does not exist, creating it")
    query = sql.SQL("CREATE DATABASE {}")
    query = query.format(sql.Identifier(db_name))
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    conn.close()

def create_database(
        if_exists:str='skip', 
    ):
    """
    Create the database and the necessary procedures.
    
    Args:
        if_exists: 'skip', 'fail' or 'override'.
        create_procedures: whether to create the procedures.
    """
    _create_database_query(if_exists=if_exists)

REQUIRED_PROCEDURES = []
@only_rank_0
def _create_all_python_procedures_query(
    project_name:str,
    conn:psycopg2.extensions.connection=None,
    ):

    commit = conn is None
    with DBConnection() if commit else conn as conn:

        # create the language for the procedures
        query = sql.SQL(
            "CREATE OR REPLACE LANGUAGE plpython3u;"
        )
        cur = conn.cursor()
        cur.execute(query)

        # create the procedure for checking that python code runs in db
        REQUIRED_PROCEDURES.append('check_python_runs')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.check_python_runs();\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.check_python_runs()\n"
            "RETURNS boolean AS $$\n"
            "return True\n"
            "$$ LANGUAGE plpython3u;"
        )
        cur = conn.cursor()
        cur.execute(query)

        # check which user is running this command
        REQUIRED_PROCEDURES.append('check_python_user')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.check_python_user();\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.check_python_user()\n"
            "RETURNS text AS $$\n"
            "import os\n"
            "return os.getenv('USER')\n"
            "$$ LANGUAGE plpython3u;"
        )   
        cur = conn.cursor()
        cur.execute(query)

        # create the procedure for getting the current working directory
        REQUIRED_PROCEDURES.append('get_psql_plpython_cwd')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.get_psql_plpython_cwd();\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.get_psql_plpython_cwd()\n"
            "RETURNS text AS $$\n"
            "import os\n"
            "return os.getcwd()\n"
            "$$ LANGUAGE plpython3u;"
        )
        cur = conn.cursor()
        cur.execute(query)

        REQUIRED_PROCEDURES.append('get_psql_plpython_username')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.get_psql_plpython_username();\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.get_psql_plpython_username()\n"
            "RETURNS text AS $$\n"
            "import os\n"
            "return os.getenv('USER')\n"
            "$$ LANGUAGE plpython3u;"
        )
        cur = conn.cursor()
        cur.execute(query)

        # create the procedure for checking if a folder exists
        REQUIRED_PROCEDURES.append('check_folder_exists')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.check_folder_exists(folder_path text);\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.check_folder_exists(folder_path text)\n"
            "RETURNS boolean AS $$\n"
            "import os\n"
            "return os.path.exists(folder_path)\n"
            "$$ LANGUAGE plpython3u;"
        )
        cur = conn.cursor()
        cur.execute(query)

        # create the procedure for removing a folder
        REQUIRED_PROCEDURES.append('remove_folder')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.remove_folder(folder_path text);\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.remove_folder(folder_path text)\n"
            "RETURNS void AS $$\n"
            "import shutil\n"
            "shutil.rmtree(folder_path)\n"
            "$$ LANGUAGE plpython3u;"
        )
        cur = conn.cursor()
        cur.execute(query)

        # create the procedure for making a folder
        REQUIRED_PROCEDURES.append('make_folder')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.make_folder(folder_path text);\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.make_folder(folder_path text)\n"
            "RETURNS void AS $$\n"
            "import os\n"
            "os.makedirs(folder_path, exist_ok=True)\n"
            "$$ LANGUAGE plpython3u;"
        )
        cur = conn.cursor()
        cur.execute(query)

        # create the procedure for checking if a file exists
        REQUIRED_PROCEDURES.append('check_file_exists')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.check_file_exists(file_path text);\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.check_file_exists(file_path text)\n"
            "RETURNS boolean AS $$\n"
            "import os\n"
            "return os.path.exists(file_path)\n"
            "$$ LANGUAGE plpython3u;"
        )
        cur = conn.cursor()
        cur.execute(query)

        # create the procedure for removing a file
        REQUIRED_PROCEDURES.append('remove_file')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.remove_file(file_path text);\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.remove_file(file_path text)\n"
            "RETURNS void AS $$\n"
            "import os\n"
            "os.remove(file_path)\n"
            "$$ LANGUAGE plpython3u;"
        )
        cur = conn.cursor()
        cur.execute(query)

        # create the procedure for listing files in a folder
        REQUIRED_PROCEDURES.append('list_files')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.list_files(folder_path text);\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.list_files(folder_path text)\n"
            "RETURNS text[] AS $$\n"
            "import os\n"
            "return os.listdir(folder_path)\n"
            "$$ LANGUAGE plpython3u;"
        )
        cur = conn.cursor()
        cur.execute(query)

        # create the procedure for appending bytes to a file
        REQUIRED_PROCEDURES.append('append_bytes_to_file')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS append_bytes_to_file(file_path text, data bytea);\n"
            f"DROP FUNCTION IF EXISTS {project_name}.append_bytes_to_file(file_path text, data bytea);\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.append_bytes_to_file(file_path text, data bytea)\n"
            "RETURNS void AS $$\n"
            "with open(file_path, 'ab') as f:  # Open in append mode\n"
            "    f.write(data)\n"
            "$$ LANGUAGE plpython3u;"
        )
        cur = conn.cursor()
        cur.execute(query)

        # create the procedure for writing bytes to a file
        REQUIRED_PROCEDURES.append('write_bytes_to_file')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.write_bytes_to_file(file_path text, data bytea);\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.write_bytes_to_file(file_path text, data bytea)\n"
            "RETURNS void AS $$\n"
            "with open(file_path, 'wb') as f:\n"
            "    f.write(data)\n"
            "$$ LANGUAGE plpython3u;"
        )
        cur = conn.cursor()
        cur.execute(query)

        # create the procedure for reading a file
        REQUIRED_PROCEDURES.append('read_bytes_from_file')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.read_bytes_from_file(file_path text, seek_n_bytes integer);\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.read_bytes_from_file(file_path text, seek_n_bytes integer)\n"
            "RETURNS bytea AS $$\n"
            "with open(file_path, 'rb') as f:\n"
            "    f.seek(seek_n_bytes)\n"
            "    return f.read()\n"
            "$$ LANGUAGE plpython3u;"
        )
        cur = conn.cursor()
        cur.execute(query)
        
        if commit:
            conn.commit()

def create_all_python_procedures(
        conn:psycopg2.extensions.connection=None,
        project_name:str=None,
    ):
    _create_all_python_procedures_query(
        conn=conn, 
        project_name=project_name,
    )

@only_rank_0
def _check_python_procedures_ok_query(
        conn:psycopg2.extensions.connection=None,
        project_name:str=None,
    ):
    commit = conn is None   
    with DBConnection() if commit else conn as conn:
        # get list of procedures that are in plpythonu3
        query = sql.SQL(
            "SELECT ns.nspname as schema_name, "
            "p.proname as function_name, "
            "pg_get_function_arguments(p.oid) as arguments "
            "FROM pg_proc p "
            "JOIN pg_namespace ns ON p.pronamespace = ns.oid "
            "JOIN pg_language l ON p.prolang = l.oid "
            "WHERE l.lanname = 'plpython3u' "
            "ORDER BY ns.nspname, p.proname;"
        )
        cur = conn.cursor()
        cur.execute(query)
        res = cur.fetchall()

        for r in res:
            if r[0] != project_name:
                continue
            if r[1] not in REQUIRED_PROCEDURES:
                raise ValueError(f"Procedure {r[1]} is not in the list of check_names!")
        print("All procedures are present!")

        query = sql.SQL(
            f"SELECT * FROM {project_name}.check_python_runs()"
        )
        cur = conn.cursor()
        cur.execute(query)
        res = cur.fetchone()[0]
    
        if not res:
            raise ValueError("Python procedures were not created successfully!")
        else:
            print("Python procedures were created successfully!")

        query = sql.SQL(
            f"SELECT * FROM {project_name}.check_python_user()"
        )
        cur = conn.cursor()
        cur.execute(query)
        res = cur.fetchone()[0]
    
def check_python_procedures_ok():
    _check_python_procedures_ok_query()

class Project:

    FILES_TABLE_NAME = 'files'
    FILES_COLUMNS = {
        'file_name': 'TEXT',
        'data': 'BYTEA',
        'type': 'TEXT',
        'sha256': 'TEXT UNIQUE',
    }
    FILES_COLUMNS_MAP = {key:i for i, key in enumerate(FILES_COLUMNS.keys())}

    def __init__(self, project_name: str, if_exists:str='skip'):
        self.db_name = get_config_item(['database', 'postgres', 'db_name'])
        assert project_name.islower(), "Project name must be in lowercase!"
        self.project_name = project_name.lower()
        self.bin_files_folder_prefix = get_config_item(['database', 'postgres', 'bin_files_folder_prefix'])
        self.project_path = os.path.abspath(os.path.join(self.bin_files_folder_prefix, self.project_name))
        self.res_prefix = os.path.join(self.project_path, 'results')
        self.ckpnt_prefix = os.path.join(self.project_path, 'checkpoints')
        self.project_name = project_name
        self.ensure_database_exists()
        self.create(if_exists=if_exists)

    def get_jobs_table_sql_identifier(self, project_name:str|None=None):
        if project_name is not None:
            project_name = project_name 
        else:
            project_name = self.project_name
        return sql.Identifier(project_name, Job.TABLE_NAME)

    def get_files_table_sql_identifier(self, project_name:str|None=None):
        if project_name is not None:
            project_name = project_name 
        else:
            project_name = self.project_name
        return sql.Identifier(project_name, Project.FILES_TABLE_NAME)

    def ensure_database_exists(self):
        _create_database_query(if_exists='skip')

    @only_rank_0
    @debounced_query
    def _exists_query(self, conn:psycopg2.extensions.connection=None):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL(
                "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = %s)"
            )
            cur = conn.cursor()
            cur.execute(query, (self.project_name,))
            res = cur.fetchone()[0]
        return res

    def exists(self):
        return self._exists_query()

    @only_rank_0
    @debounced_query
    def _drop_query(
        self,
        fail_if_not_exists: bool = False,
        cascade: bool = False,
        conn:psycopg2.extensions.connection = None, 
        ):

        commit = conn is None

        with DBConnection() if commit else conn as conn:

            folder_path = os.path.join(self.bin_files_folder_prefix, self.project_name)
            query = sql.SQL("SELECT {}.check_folder_exists(%s)")
            query = query.format(sql.Identifier(self.project_name))
            cur = conn.cursor()
            cur.execute(query, (folder_path,))
            res = cur.fetchone()[0]
            if res:
                if not cascade:
                    raise ValueError(
                        f"Folder {folder_path} does exist and it is not empty, "
                        "cannot remove it without setting cascade=True!"
                    )
                else:
                    # create the python procedures for binary files handling
                    print(
                        f"Folder {folder_path} does exist and it is not empty, "
                        "but cascade is set to True so it will be removed!"
                    )
                    query = sql.SQL("SELECT {}.remove_folder(%s)")
                    query = query.format(sql.Identifier(self.project_name))
                    cur = conn.cursor()
                    cur.execute(query, (folder_path,))

            print(f"Attempting to drop schema {self.project_name}...")
            query = sql.SQL("DROP SCHEMA {} {} {}")
            query = query.format(
                sql.SQL('IF EXISTS') if fail_if_not_exists else sql.SQL(''),
                sql.Identifier(self.project_name),
                sql.SQL('CASCADE') if cascade else sql.SQL(''),
            )
            cur = conn.cursor()
            cur.execute(query)

            if commit:
                conn.commit()

    def drop(
        self,
        fail_if_not_exists: bool = False,
        cascade: bool = False,
    ):
        self._drop_query(fail_if_not_exists=fail_if_not_exists, cascade=cascade)

    @only_rank_0
    @debounced_query
    def _create_query(
        self,
        conn:psycopg2.extensions.connection = None,
        if_exists: str = 'skip',
        ):

        commit = conn is None
        project_name = self.project_name

        with DBConnection() if commit else conn as conn:    
            if self._exists_query(conn):
                if if_exists == 'skip':
                    print(f"Schema '{project_name}' already exists, skipping creation")
                    return None
                elif if_exists == 'override':
                    self._drop_query(conn=conn, fail_if_not_exists=False, cascade=True)
                    print(f"Schema '{project_name}' already exists, but was dropped as if_exists = 'override' was set.")
                elif if_exists == 'fail':
                    raise ValueError(f"Schema '{project_name}' already exists, cannot proceed further!")
                else:
                    raise ValueError(f"Unknown value of if_exists. Expected 'skip', 'override' or 'fail', got '{if_exists}'.")

            # create schema namespace for the project
            print(f"Creating schema: '{project_name}'.")
            query = sql.SQL("CREATE SCHEMA IF NOT EXISTS {}")
            query = query.format(sql.Identifier(project_name))
            cur = conn.cursor()
            cur.execute(query)

            # create the procedures for handling files in binary files folder
            _create_all_python_procedures_query(project_name=self.project_name, conn=conn)
            _check_python_procedures_ok_query(project_name=self.project_name, conn=conn)

            # create the jobs table in the project schema
            column_specs = [
                sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(type))
                for name, type in Job.COLUMNS.items()
            ]
            query = sql.SQL("CREATE TABLE {} ({})")
            query = query.format(
                self.get_jobs_table_sql_identifier(project_name),
                sql.SQL(', ').join(column_specs)
            )
            cur = conn.cursor()
            cur.execute(query)

            # create the files table in the project schema
            column_specs = [
                sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(type))
                for name, type in Project.FILES_COLUMNS.items()
            ]
            query = sql.SQL("CREATE TABLE {} ({})")
            query = query.format(
                self.get_files_table_sql_identifier(project_name),
                sql.SQL(', ').join(column_specs)
            )
            cur = conn.cursor()
            cur.execute(query)

            # clean residual files in the binary files folder related to the project
            query = sql.SQL("SELECT {}.check_folder_exists(%s)")
            query = query.format(sql.Identifier(self.project_name))
            cur = conn.cursor()
            cur.execute(query, (self.project_path,))
            res = cur.fetchone()[0]
            if res:
                if if_exists == 'skip':
                    print(
                        f"Result folder '{self.project_path}' already exists, skipping "
                        "creation"
                    )
                elif if_exists == 'override':
                    print(
                        f"Result folder '{self.project_path}' already exists, but was "
                        "dropped as if_exists = 'override' was set."
                    )
                    query = sql.SQL("SELECT {}.remove_folder(%s)")
                    query = query.format(sql.Identifier(self.project_name))
                    cur = conn.cursor()
                    cur.execute(query, (self.project_path,))
                elif if_exists == 'fail':
                    raise ValueError(
                        f"Result folder '{self.project_path}' already exists, cannot "
                        "proceed further!"
                    )
                else:
                    raise ValueError(
                        f"Unknown value of if_exists. Expected 'skip', "
                        f"'override' or 'fail', got '{if_exists}'."
                    )

            # check cwd
            query = sql.SQL("SELECT {}.get_psql_plpython_cwd()")
            query = query.format(sql.Identifier(self.project_name))
            cur = conn.cursor()
            cur.execute(query)
            res = cur.fetchone()[0]
            print(f"Current working directory of running procedures: {res}")

            # check username
            query = sql.SQL("SELECT {}.get_psql_plpython_username()")
            query = query.format(sql.Identifier(self.project_name))
            cur = conn.cursor()
            cur.execute(query)
            res = cur.fetchone()[0]
            print(f"Username of running procedures: {res}")

            # create main folder for the project
            query = sql.SQL("SELECT {}.make_folder(%s)")
            query = query.format(sql.Identifier(self.project_name))
            cur = conn.cursor() 
            cur.execute(query, (self.project_path,))

            # create subfolder for results
            query = sql.SQL("SELECT {}.make_folder(%s)")
            query = query.format(sql.Identifier(self.project_name))
            cur = conn.cursor()
            cur.execute(query, (self.res_prefix,))

            # create subfolder for checkpoints
            query = sql.SQL("SELECT {}.make_folder(%s)")
            query = query.format(sql.Identifier(self.project_name))
            cur = conn.cursor()
            cur.execute(query, (self.ckpnt_prefix,))

            # create empty file in the project folder
            empty_file_path = os.path.join(self.project_path, f'hash_{hash_data(self.project_name)}.empty')
            query = sql.SQL("SELECT {}.write_bytes_to_file(%s, %s)")
            query = query.format(sql.Identifier(self.project_name))
            cur = conn.cursor()
            cur.execute(query, (empty_file_path, b''))

            if commit:
                conn.commit()

    def create(self, if_exists: str = 'skip'):
        self._create_query(if_exists=if_exists)
        self._set_get_result_methods_by_permissions()

    @only_rank_0
    def _set_get_result_methods_by_permissions(self):
        file_list = os.listdir(self.project_path)
        if f'hash_{hash_data(self.project_name)}.empty' not in file_list:
            print(
                f"Local {self} does not have permissions "
                f"to access results folder {self.project_path} directly.\n"
                " -> Setting psql query methods for database result reading.\n"
                "If you want to use direct read methods, you need to set "
                "the permissions to the folder manually for user that runs this code."
            )
            self.use_direct_result_read = False
        else:
            print(
                f"Local {self} does have permissions "
                f"to access results folder {self.project_path}\n"
                " -> Setting direct read methods for database result reading. Enjoy speed.."
            )
            self.use_direct_result_read = True

    @only_rank_0
    @debounced_query
    def _get_info_query(self, conn:psycopg2.extensions.connection=None):
        with DBConnection() if conn is None else conn as conn:

            query = sql.SQL(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s"
            )

            cur = conn.cursor()
            cur.execute(query, (self.project_name,))
            table_names = cur.fetchall()

            query =sql.SQL("UNION ").join([
                sql.SQL("SELECT COUNT(*) FROM {} ").format(
                    sql.Identifier(self.project_name, name[0])
                )
                for name in table_names
            ])
            cur.execute(query)
            table_data = cur.fetchall()
            res = {
                t_name[0]: t_data[0] 
                for t_name, t_data in zip(reversed(table_names), table_data)
            }
        return res

    def get_info(
            self,
            as_dataframe:bool=False,
            ):
        res = self._get_info_query()
        res = MPI.COMM_WORLD.bcast(res, root=0)
        if as_dataframe:
            res = pd.DataFrame(res, columns=['table_name'])
        return res
    
    @only_rank_0
    @debounced_query
    def _get_remote_node_name_query(
            self,
            signature:str, 
            conn:psycopg2.extensions.connection=None,
        ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL(
                "SELECT active_node_address FROM {} "
                "WHERE signature = %s"
            )
            query = query.format(self.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (signature,))
            result = cur.fetchone()[0]
        return result 
    
    def get_remote_node_name(self, signature:str):
        res = self._get_remote_node_name_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    @debounced_query
    def _set_remote_node_name_query(
            self,
            signature:str,
            active_node_address:str,
            conn:psycopg2.extensions.connection=None,
        ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL(
                "UPDATE {} SET active_node_address = %s "
                "WHERE signature = %s"
            )
            query = query.format(self.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (active_node_address, signature))
            if commit:
                conn.commit()
    
    def set_remote_node_name(self, signature:str, active_node_address:str):
        self._set_remote_node_name_query(
            signature=signature,
            active_node_address=active_node_address,
        )

    @only_rank_0
    @debounced_query
    def _get_status_query(
        self,
        signature:str,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL(
                "SELECT status FROM {} "
                "WHERE signature = %s"
            )
            query = query.format(
                self.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (signature,))
            result = cur.fetchone()[0]
        return result
    
    def get_status(self, signature:str):
        res = self._get_status_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    @debounced_query
    def _set_status_query(
        self, 
        signature:str,
        status:str,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL(
                "UPDATE {} SET status = %s, last_updated = NOW() "
                "WHERE signature = %s"
            )
            query = query.format(
                self.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (status, signature))
            if commit:
                conn.commit()

    def set_status(self, signature:str, status:str):
        self._set_status_query(signature=signature, status=status)


    @only_rank_0
    @debounced_query
    def _get_last_updated_query(self, signature:str, conn:psycopg2.extensions.connection=None):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL(
                "SELECT last_updated FROM {} "
                "WHERE signature = %s"
            )
            query = query.format(
                self.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (signature,))
            result = cur.fetchone()[0]
        return result
    
    def get_last_updated(self, signature:str):
        res = self._get_last_updated_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    @debounced_query
    def _get_error_log_query(
        self, 
        signature:str,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL(
                "SELECT error_log FROM {} "
                "WHERE signature = %s"
            )
            query = query.format(
                self.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (signature,))
            result = cur.fetchone()[0]
        return result
    
    def get_error_log(self, signature:str):
        res = self._get_error_log_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    @debounced_query
    def _set_error_log_query(
        self, 
        signature:str,
        error_log:str,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL(
                "UPDATE {} SET error_log = %s "
                "WHERE signature = %s"
            )
            query = query.format(
                self.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (error_log, signature))
            if commit:
                conn.commit()

    def set_error_log(self, signature:str, error_log:str):
        self._set_error_log_query(signature=signature, error_log=error_log)

    @only_rank_0
    @debounced_query
    def _update_progress_query(
        self, 
        signature:str,
        progress:float,
        remaining_time:int,
        elapsed_time:int,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL(
                "UPDATE {} SET "
                "progress = %s, "
                "remaining_time = %s, "
                "elapsed_time = %s, "
                "last_updated = NOW() "
                "WHERE signature = %s"
            )
            query = query.format(
                self.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (progress, remaining_time, elapsed_time, signature))
            if commit:
                conn.commit()

    def update_progress(
            self, 
            signature:str, 
            progress:float, 
            remaining_time:int, 
            elapsed_time:int,
        ):
        self._update_progress_query(
            signature=signature, 
            progress=progress, 
            remaining_time=remaining_time, 
            elapsed_time=elapsed_time,
        )

    @only_rank_0
    @debounced_query
    def _mark_interrupted_jobs_query(
        self,
        new_status:str='INTERRUPTED',
        timeout:int=MAX_WAIT_BEFORE_INTERRUPTED,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL(
                "UPDATE {} SET "
                "status = %s, "
                "last_updated = NOW(), "
                "active_node_address = 'UNASSIGNED' "
                "WHERE (status LIKE 'RUNNING - %%' OR status = 'FAILED') "
                "AND last_updated < NOW() - INTERVAL '%s seconds'"
            )
            query = query.format(
                self.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (new_status, timeout))
            if commit:
                conn.commit()
 
    def mark_interrupted_jobs(
        self, 
        new_status:str='INTERRUPTED', 
        timeout:int=MAX_WAIT_BEFORE_INTERRUPTED,
    ):
        self._mark_interrupted_jobs_query(
            new_status=new_status, 
            timeout=timeout,
        )

    @only_rank_0
    @debounced_query
    def _add_files_query(
        self, 
        files_rows:list, 
        conn:psycopg2.extensions.connection=None,
        on_conflict:str='DO NOTHING',
        ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            if on_conflict == 'DO NOTHING':
                query = sql.SQL(
                    "INSERT INTO {} ({}) VALUES %s ON CONFLICT DO NOTHING"
                ).format(
                self.get_files_table_sql_identifier(),
                sql.SQL(', ').join(
                    sql.Identifier(col) for col in Project.FILES_COLUMNS.keys()
                    ),
                )
            elif on_conflict == 'UPDATE':
                query = sql.SQL(
                    "INSERT INTO {} ({}) VALUES %s ON CONFLICT (sha256) DO UPDATE SET "
                    "file_name = EXCLUDED.file_name, "
                    "data = EXCLUDED.data, "
                    "type = EXCLUDED.type"
                ).format(
                    self.get_files_table_sql_identifier(),
                    sql.SQL(', ').join(
                        sql.Identifier(col) for col in Project.FILES_COLUMNS.keys()
                    ),
                )
            else:
                raise ValueError(f"Invalid on_conflict value: {on_conflict}! "
                                 "Must be either 'DO NOTHING' or 'UPDATE'.")
            cur = conn.cursor()
            execute_values(cur, query, files_rows)

            if commit:
                conn.commit()

    def add_files(self, files_rows:list):
        self._add_files_query(files_rows=files_rows)

    @only_rank_0
    @debounced_query
    def _check_files_exist_query(self, signatures:list[str], conn):
        with DBConnection() as conn:
            query = sql.SQL(
                "SELECT EXISTS (SELECT 1 FROM {} WHERE signature IN %s)"
            )
            query = query.format(
                self.get_files_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (tuple(signatures), ))
            res = cur.fetchone()[0]
        return res

    def check_files_exist(self, signatures:list[str]):
        res = self._check_files_exist_query(signatures=signatures)
        return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    @debounced_query
    def _get_files_query(
        self, 
        conn:psycopg2.extensions.connection = None, 
        signatures:list[str] | None = None,
        ):
        with DBConnection() if conn is None else conn as conn:
            if signatures is not None:
                query = sql.SQL("SELECT * FROM {} WHERE sha256 IN %s")
            else:
                query = sql.SQL("SELECT * FROM {}")
            query = query.format(
                self.get_files_table_sql_identifier()
            )
            cur = conn.cursor()
            if signatures is not None:
                cur.execute(query, (tuple(signatures), ))
            else:
                cur.execute(query)
            rows = cur.fetchall()
            rows = [{
                'file_name': row[0], 
                'data': row[1].tobytes(), 
                'type': row[2], 
                'sha256': row[3]
                } for row in rows]
        return rows
    
    def get_files(
            self, 
            signatures:list[str]=None, 
            as_dataframe:bool=False,
        ):
        rows = self._get_files_query(signatures=signatures)
        rows = MPI.COMM_WORLD.bcast(rows, root=0)
        if as_dataframe:
            return pd.DataFrame(rows, columns=Project.FILES_COLUMNS.keys())
        else:
            return rows
    
    @only_rank_0
    @debounced_query
    def _add_jobs_query(
            self, 
            jobs:list,
            conn:psycopg2.extensions.connection = None,
            ):

        commit = conn is None
        with DBConnection() if commit else conn as conn:
            page_size = 1000

            # insert jobs
            jobs_query = sql.SQL(
                "INSERT INTO {} ({}) VALUES %s ON CONFLICT DO NOTHING"
            ).format(
                self.get_jobs_table_sql_identifier(),
                sql.SQL(', ').join(
                    sql.Identifier(col) for col in Job.COLUMNS.keys()
                ),
            )
            cur = conn.cursor()
            job_rows = []
            unique_source_files_rows = set()
            for job in jobs:

                # jobs table data copy not to mess with the original data
                cp_data = job.data.copy()

                # insert and last updated datetime
                cp_data['insert_datetime'] = str(datetime.datetime.now(datetime.timezone.utc))
                cp_data['last_updated'] = cp_data['insert_datetime']

                # dump p_inputs into database compatible jsonb format
                cp_data['p_inputs'] = json.dumps(cp_data['p_inputs'])
                job_rows.append([cp_data[key] for key in Job.COLUMNS.keys()])

                # files table data
                for row in job.source_files_data:
                    hashable_row = tuple(row.values())
                    if hashable_row not in unique_source_files_rows:
                        unique_source_files_rows.add(hashable_row)

            execute_values(cur, jobs_query, job_rows, page_size=page_size)
            self._add_files_query(files_rows=list(unique_source_files_rows))

            if commit:
                conn.commit()

    def add_jobs(self, jobs):
        self._add_jobs_query(jobs=jobs)

    @only_rank_0
    @debounced_query
    def _add_meshes_query(
        self,
        dir:str,
        names:list[str], 
        conn:psycopg2.extensions.connection = None,
    ):

        commit = conn is None
        files_rows = []
        for name in names:
            for ext in ['msh', 'ad', 'step']:
                file_name = ".".join([name, ext])
                file_path = os.path.join(dir, file_name)
                with open(file_path, 'rb') as f:
                    data = f.read()
                    files_rows.append(
                        (file_name, data, f'mesh_{ext}', hash_data(data))
                    )
        with DBConnection() if commit else conn as conn:
            self._add_files_query(files_rows=files_rows, conn=conn)

            if commit:
                conn.commit()

    def add_meshes(self, dir:str, names:list[str]):
        self._add_meshes_query(dir=dir, names=names)

    @only_rank_0
    @debounced_query
    def _get_meshes_query(
        self, 
        dir:str,
        names:list[str], 
        conn:psycopg2.extensions.connection=None,
    ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("SELECT * FROM {} WHERE file_name IN %s")
            query = query.format(
                self.get_files_table_sql_identifier(),
            )
            cur = conn.cursor()

            for name in names:
                files_names = []
                for ext in ['msh', 'ad', 'step']:
                    files_names.append(".".join([name, ext]))
                cur.execute(query, (tuple(files_names), ))
                data_rows = cur.fetchall()
                for data_row in data_rows:
                    file_name = data_row[Project.FILES_COLUMNS_MAP['file_name']]
                    os.makedirs(dir, exist_ok=True)
                    with open(os.path.join(dir, file_name), 'wb') as f:
                        f.write(data_row[Project.FILES_COLUMNS_MAP['data']])
    
    def get_meshes(self, names:list[str], dir:str):
        res = self._get_meshes_query(dir=dir, names=names)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    @debounced_query
    def _check_meshes_exist_query(
        self, 
        names:list[str], 
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL(
                "SELECT EXISTS (SELECT 1 FROM {} WHERE file_name IN %s)"
            )
            query = query.format(
                self.get_files_table_sql_identifier(),
            )
            cur = conn.cursor()
            files_names = []
            for name in names:
                for ext in ['msh', 'ad', 'step']:
                    files_names.append(".".join([name, ext]))
            cur.execute(query, (tuple(files_names), ))
            res = cur.fetchone()[0]
        return res

    def check_meshes_exist(self, names:list[str]):
        res = self._check_meshes_exist_query(names=names)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    # @only_rank_0
    # @debounced_query
    # def _add_meshes_query_v2(
    #     self,
    #     dir:str,
    #     names:list[str],
    #     conn:psycopg2.extensions.connection=None,
    # ):
    #     commit = conn is None

    #     for name in names:
    #         memory_file = io.BytesIO()
    #         with tarfile.open(fileobj=memory_file, mode='w') as tar:
    #             for ext in ['msh', 'ad', 'step']:
    #                 file_name = ".".join([name, ext])
    #                 file_path = os.path.join(dir, file_name)
    #                 tar.add(file_path, arcname=file_name)
        
    #     memory_file.seek(0)
    #     data = memory_file.getvalue()

    #     with DBConnection() if conn is None else conn as conn:
    #         query = sql.SQL("SELECT {}.write_bytes_to_file(%s, %s)")
    #         query = query.format(sql.Identifier(self.project_name))
    #         cur = conn.cursor()
    #         cur.execute(query, (f'mesh_{name}', data))

    #         if commit:
    #             conn.commit()


    # def add_meshes_v2(self, dir:str, names:list[str]):
    #     self._add_meshes_query_v2(dir=dir, names=names)

    # @only_rank_0
    # @debounced_query
    # def _get_meshes_query_v2(
    #     self,
    #     dir:str,
    #     names:list[str],
    #     conn:psycopg2.extensions.connection=None,
    # ):
    #     with DBConnection() if conn is None else conn as conn:
    #         query = sql.SQL("SELECT {}.read_bytes_from_file(%s, %s)")
    #         query = query.format(sql.Identifier(self.project_name))
    #         cur = conn.cursor()
    #         for name in names:
    #             cur.execute(query, (f'mesh_{name}', 0))
    #             data = cur.fetchone()[0]
    #             memory_file = io.BytesIO(data)
    #             with tarfile.open(fileobj=memory_file, mode='r') as tar:
    #                 tar.extractall(path=dir)

    # def get_meshes_v2(self, dir:str, names:list[str]):
    #     res = self._get_meshes_query_v2(dir=dir, names=names)
    #     return MPI.COMM_WORLD.bcast(res, root=0)
    
    # @only_rank_0
    # @debounced_query
    # def _check_meshes_exist_query_v2(
    #     self,
    #     names:list[str],
    #     conn:psycopg2.extensions.connection=None,
    # ):
    #     with DBConnection() if conn is None else conn as conn:
    #         query = sql.SQL("SELECT {}.check_file_exists(%s)")
    #         query = query.format(sql.Identifier(self.project_name))
    #         cur = conn.cursor()
    #         for name in names:
    #             cur.execute(query, (f'mesh_{name}', ))
    #             res = cur.fetchone()[0]
    #     return res
    
    # def check_meshes_exist_v2(self, names:list[str]):
    #     res = self._check_meshes_exist_query_v2(names=names)
    #     return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    @debounced_query
    def _add_checkpoint_query(
            self, 
            signature:str, 
            checkpoint_dir:str,
            conn:psycopg2.extensions.connection=None,
        ):
        commit = conn is None

        # read metadata
        metadata_file = os.path.join(checkpoint_dir, "metadata.json")
        with open(metadata_file, 'rb') as fp:
            metadata = json.load(fp)

        # Create zip archive in memory
        #memory_file = io.BytesIO()
        memory_file = io.BytesIO()
        with tarfile.open(fileobj=memory_file, mode='w') as tar:
            # Add all files from checkpoint directory to tar
            tar.add(checkpoint_dir, arcname='')
        
        # Get the bytes from the memory file and push to database as a file
        memory_file.seek(0)
        data = memory_file.getvalue()
        with DBConnection() if conn is None else conn as conn:

            # upload checkpoint to database
            #self._add_files_query(files_rows=files_rows, conn=conn, on_conflict='UPDATE')
            binary_file_name = os.path.join(
                self.ckpnt_prefix,  
                f'{signature}.chkpnt'
            )
            query = sql.SQL("SELECT {}.write_bytes_to_file(%s, %s)")
            query = query.format(sql.Identifier(self.project_name))
            cur = conn.cursor()
            cur.execute(query, (binary_file_name, data))

            # update job table
            query = sql.SQL(
                "UPDATE {} SET "
                "last_checkpoint_progress = %s, "
                "last_checkpoint_date = NOW() "
                "WHERE signature = %s"
            )
            query = query.format(
                self.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (
                metadata['progress'], 
                #metadata['local_timestamp'], 
                signature)
            )

            if commit:
                conn.commit()

    def add_checkpoint(self, signature:str, checkpoint_dir:str):
        self._add_checkpoint_query(
            signature=signature, 
            checkpoint_dir=checkpoint_dir,
        )

    @only_rank_0
    @debounced_query
    def _get_checkpoint_query(
        self, 
        checkpoint_dir:str,
        signature:str,
        conn:psycopg2.extensions.connection=None,
    ):
        with DBConnection() if conn is None else conn as conn:
            # query = sql.SQL("SELECT * FROM {} WHERE file_name = %s AND type = 'checkpoint' AND sha256 = %s")
            # query = query.format(
            #     self.get_files_table_sql_identifier(),
            # )
            # cur = conn.cursor()
            # cur.execute(query, (f'ckpnt_{signature}', signature))
            # file_row = cur.fetchone()
            # data = file_row[Project.FILES_COLUMNS_MAP['data']]
            # memory_file = io.BytesIO(data)

            binary_file_name = os.path.join(
                self.ckpnt_prefix,  
                f'{signature}.chkpnt'
            )

            query = sql.SQL("SELECT {}.read_bytes_from_file(%s, %s)")
            query = query.format(sql.Identifier(self.project_name))
            cur = conn.cursor()
            cur.execute(query, (binary_file_name, 0))
            data = cur.fetchone()[0]
            memory_file = io.BytesIO(data)
            with tarfile.open(fileobj=memory_file, mode='r') as tar:
                tar.extractall(path=checkpoint_dir)
    
    def get_checkpoint(self, signature:str, checkpoint_dir:str):
        self._get_checkpoint_query(
            signature=signature, 
            checkpoint_dir=checkpoint_dir,
        )

    @only_rank_0
    @debounced_query
    def _get_checkpoint_info_query(
        self, 
        signature:str, 
        conn:psycopg2.extensions.connection=None,
    ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("SELECT last_checkpoint_progress, last_checkpoint_date FROM {} WHERE signature = %s")
            query = query.format(self.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (signature, ))
            res = cur.fetchone()
            res = {'progress': res[0], 'local_timestamp': res[1]}
        return res
    
    def get_checkpoint_info(self, signature:str):
        res = self._get_checkpoint_info_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    @debounced_query
    def _check_checkpoint_exist_query(
        self, 
        signature:str,
        conn:psycopg2.extensions.connection=None,
    ):
        with DBConnection() if conn is None else conn as conn:
            binary_file_name = os.path.join(
                self.ckpnt_prefix,  
                f'{signature}.chkpnt'
            )
            query = sql.SQL("SELECT {}.check_file_exists(%s)")
            query = query.format(sql.Identifier(self.project_name))
            cur = conn.cursor()
            cur.execute(query, (binary_file_name, ))
            res = cur.fetchone()[0]
        return res

    def check_checkpoint_exist(self, signature:str):
        res = self._check_checkpoint_exist_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    @debounced_query
    def _get_jobs_query(
        self, 
        conn:psycopg2.extensions.connection=None, 
        ):
        if MPI.COMM_WORLD.rank == 0:
            with DBConnection() if conn is None else conn as conn:
                query = sql.SQL(
                    "SELECT * FROM {}"
                )
                query = query.format(
                    self.get_jobs_table_sql_identifier(),
                    )
                cur = conn.cursor()
                cur.execute(query)
                jobs_rows = cur.fetchall()
            return jobs_rows

    def get_jobs(
            self, 
            as_dataframe:bool=False,
        ):
        jobs_rows = self._get_jobs_query()
        jobs_rows = MPI.COMM_WORLD.bcast(jobs_rows, root=0)
        if as_dataframe:
            return pd.DataFrame(jobs_rows, columns=Job.COLUMNS.keys())
        else:
            return [
                Job(dict(zip(Job.COLUMNS.keys(), row)), [], self) 
                for row in jobs_rows
            ]  

    @only_rank_0
    @debounced_query
    def _get_parameter_space_query(
        self, 
        group_names:list[str]|None=None,
        conn:psycopg2.extensions.connection=None    
    ):
        with DBConnection() if conn is None else conn as conn:
            if group_names is None:
                query = sql.SQL("SELECT signature, group_signature, group_name, p_inputs FROM {}")
                query = query.format(self.get_jobs_table_sql_identifier())
            else:
                query = sql.SQL("SELECT signature, group_signature, group_name, p_inputs FROM {} WHERE group_name IN %s")
                query = query.format(self.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (tuple(group_names), ))
            res = cur.fetchall()
        return res
    
    def get_parameter_space(
            self, 
            group_name:str|None=None,
        ):
        res = self._get_parameter_space_query(group_name=group_name)
        return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    @debounced_query
    def _get_next_scheduled_job_query(
        self, 
        conn:psycopg2.extensions.connection=None,
        ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL(
                "SELECT * FROM {} WHERE "
                "status = 'SCHEDULED' OR "
                "status LIKE 'FAILED%%' OR "
                "status = 'INTERRUPTED' "
                "ORDER BY priority ASC LIMIT 1"
            )  
            query = query.format(
                self.get_jobs_table_sql_identifier(),
                )
            cur = conn.cursor()
            cur.execute(query)
            row = cur.fetchone()

            required_source_files = row[Job.COLUMNS_MAP['required_source_files']]
            source_files_rows = self._get_files_query(
                conn=conn, 
                signatures=required_source_files,
            )
        if row is not None:
            return Job(dict(zip(Job.COLUMNS.keys(), row)), source_files_rows, self)

    def get_next_scheduled_job(self) -> Job | None:
        res = self._get_next_scheduled_job_query()
        res = MPI.COMM_WORLD.bcast(res, root=0)
        MPI.COMM_WORLD.barrier()
        return res

    @only_rank_0
    @debounced_query
    def _get_job_query(
        self, 
        signature:str, 
        conn:psycopg2.extensions.connection=None, 
        ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL(
                "SELECT * FROM {} WHERE signature = %s"
            )
            query = query.format(
                self.get_jobs_table_sql_identifier(self.project_name),
                )
            cur = conn.cursor()
            cur.execute(query, (signature,))
            row = cur.fetchone()

            required_source_files = row[Job.COLUMNS_MAP['required_source_files']]
            source_files_rows = self._get_files_query(
                conn=conn, 
                signatures=required_source_files,
            )
        if row is not None:
            return Job(dict(zip(Job.COLUMNS.keys(), row)), source_files_rows, self)
        
    def get_job(self, signature:str):
        res = self._get_job_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    def _get_interupted_jobs_query(self, conn:psycopg2.extensions.connection=None):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("SELECT * FROM {} WHERE status != 'COMPLETED' AND last_updated < NOW() - INTERVAL '10 minutes'")
            query = query.format(
                self.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query)
            res_rows = cur.fetchall()
            return [Job(dict(zip(Job.COLUMNS.keys(), row)), [], self) for row in res_rows]   

    def get_interupted_jobs(self):
        res = self._get_interupted_jobs_query()
        return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    @debounced_query
    def _get_result_table_signatures_query(
        self, 
        conn:psycopg2.extensions.connection=None,
        ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("SELECT {}.list_files(%s)")
            query = query.format(sql.Identifier(self.project_name))
            cur = conn.cursor()
            cur.execute(query, (self.res_prefix, ))
            files_res = [r.split('.')[0] for r in cur.fetchone()[0] if r.endswith('.hbres')]

            # # this might be not necessary, but just to be sure..
            # query = sql.SQL("SELECT signature FROM {} WHERE progress > 0")
            # query = query.format(self.get_jobs_table_sql_identifier())
            # cur.execute(query)
            # jobs_res_signatures = [r[0] for r in cur.fetchall()]

            # intersection = [fr for fr in files_res if fr in jobs_res_signatures]
            return files_res
    
    def get_result_table_signatures(self, as_dataframe:bool=False):
        res = self._get_result_table_signatures_query()
        res = MPI.COMM_WORLD.bcast(res, root=0)
        if as_dataframe:
            return pd.DataFrame(res, columns=['table_name'])
        else:
            return res
        
    @only_rank_0
    @debounced_query
    def _get_result_table_columns_query(
        self, 
        signature:str,
        conn:psycopg2.extensions.connection=None,  
        ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("SELECT probe_columns FROM {} WHERE signature = %s")
            query = query.format(sql.Identifier(self.project_name, 'jobs'))
            cur = conn.cursor()
            cur.execute(query, (signature, ))
            res = cur.fetchone()[0]
        return res

    def get_result_table_columns(self, signature:str):
        res = self._get_result_table_columns_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    @debounced_query
    def _create_result_binary_file_query(
        self,
        signature:str,
        columns:list[str],
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if conn is None else conn as conn:

            #header data
            col_names_bytes = bytes('\n'.join(columns), 'utf-8')
            n_cols_bytes = np.int64(len(columns)).tobytes()
            n_cols_names_bytes = np.int64(len(col_names_bytes)).tobytes()
            header_bytes = n_cols_bytes + n_cols_names_bytes + col_names_bytes

            #call pyfunction that creates the file
            binary_file_name = os.path.join(
                self.res_prefix,  
                f'{signature}.hbres'
            )
            query = sql.SQL(
                "SELECT {}.write_bytes_to_file(%s, %s)"
            )
            query = query.format(sql.Identifier(self.project_name))
            cur = conn.cursor()
            cur.execute(query, (binary_file_name, header_bytes))
            if commit:
                conn.commit()

    def create_result_binary_file(self, signature:str):
        self._create_result_binary_file_query(signature=signature)

    @only_rank_0
    @debounced_query
    def _append_data_to_result_binary_file_query(
            self, 
            signature:str, 
            data:bytes,
            conn:psycopg2.extensions.connection=None,
        ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("SELECT {}.append_bytes_to_file(%s, %s)")
            query = query.format(sql.Identifier(self.project_name))
            binary_file_path = os.path.join(
                self.res_prefix, 
                f'{signature}.hbres'
            )
            cur = conn.cursor()
            cur.execute(query, (binary_file_path, data))

    def append_data_to_result_binary_file(self, signature:str, data:bytes):
        self._append_data_to_result_binary_file_query(signature=signature, data=data)

    @only_rank_0
    @debounced_query
    def _get_result_binary_file_query(
        self, 
        signature:str,
        skip_n_bytes:int=0,
        conn:psycopg2.extensions.connection=None,
    ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("SELECT {}.read_bytes_from_file(%s, %s)")
            query = query.format(
                sql.Identifier(self.project_name),
            )
            path = os.path.join(
                self.res_prefix, 
                f'{signature}.hbres'
            )
            cur = conn.cursor()
            cur.execute(query, (path, skip_n_bytes))
            res = cur.fetchone()[0] #this should by bytes
        return res

    def get_result_binary_file(self, signature:str, skip_n_bytes:int=0):
        res = self._get_result_binary_file_query(
            signature=signature,
            skip_n_bytes=skip_n_bytes,
        )
        return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    def _parse_binary_result_bytes(self, binary_data:bytes):
        size = len(binary_data)
        mem_view_bytes = memoryview(binary_data)
        n_cols = np.frombuffer(mem_view_bytes[:8], dtype=np.int64)[0]
        n_cols_names = np.frombuffer(mem_view_bytes[8:16], dtype=np.int64)[0]
        col_names = mem_view_bytes[16:16+n_cols_names].tobytes().decode('utf-8').split('\n')
        assert ((size-16-n_cols_names) % (n_cols * 8) == 0), (
            "Size of binary data is not divisible by number size of float64 (8 bytes)"
        )
        n_rows = int((size-16-n_cols_names) / (n_cols * 8))

        start_time = time.time()
        array = np.frombuffer(mem_view_bytes[16+n_cols_names:], dtype=np.float64).reshape(n_rows, n_cols)
        return array, col_names, n_cols, n_rows
        
    @only_rank_0
    @debounced_query
    def _get_result_numpy_array_query(
        self, 
        signature:str, 
        skip_n_bytes:int=0,
        conn:psycopg2.extensions.connection=None,
    ):
        with DBConnection() if conn is None else conn as conn:
            binary_data = self._get_result_binary_file_query(
                signature=signature,
                skip_n_bytes=skip_n_bytes,
            )
            parsed_data = self._parse_binary_result_bytes(binary_data)
        return parsed_data[0]
    
    def get_result_numpy_array(self, signature:str, skip_n_bytes:int=0):
        res = self._get_result_numpy_array_query(
            signature=signature, 
            skip_n_bytes=skip_n_bytes,
        )
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    def _get_result_binary_file_direct(
            self, 
            signature:str, 
            skip_n_bytes:int=0,
        ):
        path = os.path.join(
            self.res_prefix, 
            f'{signature}.hbres'
            )
        with open(path, 'rb') as f:
            f.seek(skip_n_bytes)
            binary_data = f.read()
        return binary_data
    
    def get_result_binary_file_direct(
            self, 
            signature:str, 
            skip_n_bytes:int=0,
        ):
        res = self._get_result_binary_file_direct(
            signature=signature, 
            skip_n_bytes=skip_n_bytes,
        )
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    def _get_result_numpy_array_direct(
            self, 
            signature:str, 
            skip_n_bytes:int=0,
        ):
        binary_data = self._get_result_binary_file_direct(
            signature=signature, 
            skip_n_bytes=skip_n_bytes,
        )
        parsed_data = self._parse_binary_result_bytes(binary_data)
        return parsed_data[0]
    
    def get_result_numpy_array_direct(
            self, 
            signature:str, 
            skip_n_bytes:int=0,
        ):
        res = self._get_result_numpy_array_direct(
            signature=signature, 
            skip_n_bytes=skip_n_bytes,
        )
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    @debounced_query
    def _get_result_dataframe_query(
            self, 
            signature:str, 
            skip_n_bytes:int=0,
        ):
        data = self._get_result_binary_file_query(
            signature=signature, 
            skip_n_bytes=skip_n_bytes,
        )
        array, col_names, n_cols, n_rows = self._parse_binary_result_bytes(data)
        return pd.DataFrame(array, columns=col_names)
    
    @only_rank_0
    def _get_result_dataframe_direct(
            self, 
            signature:str, 
            skip_n_bytes:int=0,
        ):
        data = self._get_result_binary_file_direct(
            signature=signature, 
            skip_n_bytes=skip_n_bytes,  
        )
        array, col_names, n_cols, n_rows = self._parse_binary_result_bytes(data)
        return pd.DataFrame(array, columns=col_names)
    
    def get_result_dataframe_direct(
            self, 
            signature:str, 
            skip_n_bytes:int=0,
        ):
        res = self._get_result_dataframe_direct(
            signature=signature, 
            skip_n_bytes=skip_n_bytes,
        )
        return MPI.COMM_WORLD.bcast(res, root=0)

    def _get_result_binary_file(self, signature:str, skip_n_bytes:int=0):
        if self.use_direct_result_read:
            return self._get_result_binary_file_direct(signature=signature, skip_n_bytes=skip_n_bytes)
        else:
            return self._get_result_binary_file_query(signature=signature, skip_n_bytes=skip_n_bytes)
        
    def _get_result_numpy_array(self, signature:str, skip_n_bytes:int=0):
        if self.use_direct_result_read:
            return self._get_result_numpy_array_direct(signature=signature, skip_n_bytes=skip_n_bytes)
        else:
            return self._get_result_numpy_array_query(signature=signature, skip_n_bytes=skip_n_bytes)
        
    def _get_result_dataframe(self, signature:str, skip_n_bytes:int=0):
        if self.use_direct_result_read:
            return self._get_result_dataframe_direct(signature=signature, skip_n_bytes=skip_n_bytes)
        else:
            return self._get_result_dataframe_query(signature=signature, skip_n_bytes=skip_n_bytes)

    @only_rank_0
    @debounced_query
    def _upload_result_table_query(
        self, 
        signature:str,
        df:pd.DataFrame,
        conn:psycopg2.extensions.connection=None,
        ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            self._create_result_binary_file_query(signature=signature, columns=df.columns, conn=conn)
            self._append_data_to_result_binary_file_query(signature=signature, data=df.to_numpy(dtype=np.float64).tobytes(), conn=conn)

    def upload_result_table(
            self, 
            signature:str, 
            df:pd.DataFrame, 
        ):
        self._upload_result_table_query(signature=signature, df=df)

    # @only_rank_0
    # @debounced_query
    # def _copy_csv_to_result_table_query(
    #     self, 
    #     signature:str, 
    #     path_to_csv:str,
    #     conn:psycopg2.extensions.connection=None,
    #     ):
    #     commit = conn is None
    #     with DBConnection() if commit else conn as conn:
    #         query = sql.SQL("COPY {} FROM {} WITH CSV HEADER")
    #         query = query.format(
    #             sql.Identifier(self.project_name, f'res_{signature}'),
    #             sql.Literal(path_to_csv),
    #         )
    #         cur = conn.cursor()
    #         cur.execute(query)
    #         if commit:
    #             conn.commit()

    # def copy_csv_to_result_table(
    #         self, 
    #         signature:str, 
    #         path_to_csv:str,
    #     ):
    #     self._copy_csv_to_result_table_query(
    #         signature=signature, 
    #         path_to_csv=path_to_csv,
    #     )

    # @only_rank_0
    # @debounced_query
    # def _clean_files_query(
    #     self, 
    #     conn:psycopg2.extensions.connection=None, 
    #     ):
    #     with DBConnection() if conn is None else conn as conn:
    #         query = sql.SQL(
    #             "SELECT array_agg(uq) FROM "
    #             "(SELECT DISTINCT unnest(required_source_files) as uq FROM {})"
    #             "subquery"
    #         )
    #         query = query.format(
    #             self.get_jobs_table_sql_identifier(self.project_name),
    #             )
    #         cur =  conn.cursor()
    #         cur.execute(query)
    #         res = cur.fetchone()[0]
    #     return res
    
    # def clean_files(self):
    #     res = self._clean_files_query()
    #     return MPI.COMM_WORLD.bcast(res, root=0)
 
    def __repr__(self):
        return f"Project(name={self.project_name})"

@only_rank_0
def _list_remote_projects_query(conn:psycopg2.extensions.connection=None):
    with DBConnection() if conn is None else conn as conn:
        query = sql.SQL(
            "SELECT nspname FROM pg_namespace "
            "WHERE nspname NOT IN ('pg_toast', 'pg_catalog', 'public', 'information_schema')"
        )
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        res = [Project(row[0]) for row in rows]
        return res
    
def list_remote_projects():
    res = _list_remote_projects_query()
    return MPI.COMM_WORLD.bcast(res, root=0)

@only_rank_0
def _remove_all_remote_projects_query(conn:psycopg2.extensions.connection=None):
    commit = conn is None
    with DBConnection() if conn is None else conn as conn:
        projects = _list_remote_projects_query(conn=conn)
        for project in projects:
            project._drop_query(
                fail_if_not_exists=True,
                cascade=True,
                conn=conn,
            )
        if commit:
            conn.commit()

def remove_all_remote_projects():
    _remove_all_remote_projects_query()