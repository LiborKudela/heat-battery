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

MAX_WAIT_BEFORE_INTERRUPTED = 600 # seconds

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
            "try:\n"
            "    import subprocess\n"
            "    result = subprocess.run(['whoami'], stdout=subprocess.PIPE, text=True)\n"
            "    return result.stdout.strip()\n"
            "except Exception as e:\n"
            "    return str(e)\n"
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

        # create the procedure for truncating a file at a specific position
        REQUIRED_PROCEDURES.append('truncate_file')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.truncate_file(file_path text, keep_n_bytes integer);\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.truncate_file(file_path text, keep_n_bytes integer)\n"
            "RETURNS void AS $$\n"
            "with open(file_path, 'r+b') as f:\n"
            "    f.seek(keep_n_bytes)\n"
            "    f.truncate()\n"
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

        # create the procedure for getting the size of a file in bytes
        REQUIRED_PROCEDURES.append('get_file_size')
        query = sql.SQL(
            f"DROP FUNCTION IF EXISTS {project_name}.get_file_size(file_path text);\n"
            f"CREATE OR REPLACE FUNCTION {project_name}.get_file_size(file_path text)\n"
            "RETURNS integer AS $$\n"
            "import os\n"
            "return os.path.getsize(file_path)\n"
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

    BACKUP_VERSION = 1
    BACKUP_MANIFEST_NAME = 'manifest.json'
    BACKUP_JOBS_NAME = 'jobs.bin'
    BACKUP_FILES_NAME = 'files.bin'
    BACKUP_RESULTS_PREFIX = 'results/'
    BACKUP_CHECKPOINTS_PREFIX = 'checkpoints/'

    @staticmethod
    def _add_bytes_to_tar(tar:tarfile.TarFile, arcname:str, data:bytes):
        info = tarfile.TarInfo(name=arcname)
        info.size = len(data)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(data))

    @only_rank_0
    @debounced_query
    def _save_to_file_query(
        self,
        file_path:str,
        conn:psycopg2.extensions.connection=None,
    ):
        """
        Dump everything that defines the current state of this Project (database
        tables under the ``self.project_name`` schema + all binary files stored
        in the result and checkpoint folders) into a single ``.tar.gz`` archive.
        """
        commit = conn is None
        file_path = os.path.abspath(file_path)
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with DBConnection() if commit else conn as conn:
            with tarfile.open(file_path, mode='w:gz') as tar:

                # 1) Manifest with metadata
                manifest = {
                    'version': Project.BACKUP_VERSION,
                    'project_name': self.project_name,
                    'db_name': self.db_name,
                    'created_at': str(datetime.datetime.now(datetime.timezone.utc)),
                    'jobs_columns': list(Job.COLUMNS.keys()),
                    'files_columns': list(Project.FILES_COLUMNS.keys()),
                }
                manifest_bytes = json.dumps(manifest, indent=2).encode('utf-8')
                self._add_bytes_to_tar(
                    tar, Project.BACKUP_MANIFEST_NAME, manifest_bytes,
                )

                # 2) Dump 'jobs' table in PostgreSQL binary format (lossless)
                print_rank_0(f"Dumping table {self.project_name}.{Job.TABLE_NAME}...")
                jobs_buf = io.BytesIO()
                copy_jobs_q = sql.SQL("COPY {} TO STDOUT WITH BINARY").format(
                    self.get_jobs_table_sql_identifier(),
                )
                cur = conn.cursor()
                cur.copy_expert(copy_jobs_q.as_string(conn), jobs_buf)
                self._add_bytes_to_tar(
                    tar, Project.BACKUP_JOBS_NAME, jobs_buf.getvalue(),
                )

                # 3) Dump 'files' table in PostgreSQL binary format (lossless)
                print_rank_0(f"Dumping table {self.project_name}.{Project.FILES_TABLE_NAME}...")
                files_buf = io.BytesIO()
                copy_files_q = sql.SQL("COPY {} TO STDOUT WITH BINARY").format(
                    self.get_files_table_sql_identifier(),
                )
                cur = conn.cursor()
                cur.copy_expert(copy_files_q.as_string(conn), files_buf)
                self._add_bytes_to_tar(
                    tar, Project.BACKUP_FILES_NAME, files_buf.getvalue(),
                )

                # 4) Dump on-disk binary files (results and checkpoints) using
                #    the python procedures so it works even when the local user
                #    has no permission on the server's filesystem.
                list_files_q = sql.SQL("SELECT {}.list_files(%s)").format(
                    sql.Identifier(self.project_name),
                )
                read_q = sql.SQL("SELECT {}.read_bytes_from_file(%s, %s)").format(
                    sql.Identifier(self.project_name),
                )

                for label, folder, prefix in (
                    ('result',     self.res_prefix,   Project.BACKUP_RESULTS_PREFIX),
                    ('checkpoint', self.ckpnt_prefix, Project.BACKUP_CHECKPOINTS_PREFIX),
                ):
                    cur = conn.cursor()
                    cur.execute(list_files_q, (folder,))
                    file_names = cur.fetchone()[0] or []
                    print_rank_0(
                        f"Dumping {len(file_names)} {label} file(s) from {folder}..."
                    )
                    for fname in file_names:
                        remote_path = os.path.join(folder, fname)
                        cur = conn.cursor()
                        cur.execute(read_q, (remote_path, 0))
                        raw = cur.fetchone()[0]
                        data = bytes(raw) if raw is not None else b''
                        self._add_bytes_to_tar(tar, prefix + fname, data)

            if commit:
                conn.commit()

        return file_path

    def save_to_file(self, file_path:str):
        """
        Export this project's full state to a single ``.tar.gz`` archive that
        can later be restored with :meth:`load_from_file`.

        The archive contains:

        * ``manifest.json``              - project metadata and original name.
        * ``jobs.bin``                   - PostgreSQL binary dump of the jobs table.
        * ``files.bin``                  - PostgreSQL binary dump of the files table.
        * ``results/<signature>.hbres``  - per-job result binary files.
        * ``checkpoints/<sig>.chkpnt``   - per-job checkpoint archives.

        Args:
            file_path: Path to the output archive (``.tar.gz`` recommended).

        Returns:
            Absolute path of the produced archive (only on rank 0; ``None`` on
            other ranks - use :func:`mpi4py.MPI.COMM_WORLD.bcast` if needed).
        """
        return self._save_to_file_query(file_path=file_path)

    @only_rank_0
    @debounced_query
    def _load_from_file_query(
        self,
        file_path:str,
        recreate:bool=True,
        conn:psycopg2.extensions.connection=None,
    ):
        """
        Restore project state from an archive produced by :meth:`save_to_file`
        into the schema ``self.project_name``. The original project name is
        ignored (taken only as informational metadata), so a backup of
        ``project_example_05`` can be loaded into ``project_example_05_copy``.
        """
        commit = conn is None
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Backup archive '{file_path}' does not exist!"
            )

        # Drop and recreate the project schema and folders in their own
        # transaction so we always load into a clean state. We deliberately
        # do NOT pass `conn` here: `_create_query` internally re-enters
        # `with conn as conn` and so does `_exists_query`/`_drop_query`,
        # which psycopg2 forbids on an already-entered connection
        # ("the connection cannot be re-entered recursively").
        # Doing this first also means the destructive drop is committed
        # before we start streaming bulk data into the new schema.
        if recreate:
            print_rank_0(
                f"Recreating clean schema '{self.project_name}' before "
                "restoring backup..."
            )
            self._create_query(if_exists='override')

        with DBConnection() if commit else conn as conn:

            with tarfile.open(file_path, mode='r:*') as tar:

                # 1) Manifest
                manifest_member = tar.getmember(Project.BACKUP_MANIFEST_NAME)
                manifest = json.loads(
                    tar.extractfile(manifest_member).read().decode('utf-8'),
                )
                if manifest.get('version') != Project.BACKUP_VERSION:
                    raise ValueError(
                        f"Unsupported backup version: {manifest.get('version')!r} "
                        f"(this code expects version {Project.BACKUP_VERSION})."
                    )
                src_name = manifest.get('project_name', '<unknown>')
                print_rank_0(
                    f"Restoring backup of project '{src_name}' "
                    f"(created at {manifest.get('created_at', '?')}) "
                    f"into '{self.project_name}'..."
                )

                # Sanity check: schemas must match the destination's columns.
                if manifest.get('jobs_columns') != list(Job.COLUMNS.keys()):
                    raise ValueError(
                        "Backup 'jobs' columns do not match current Job.COLUMNS:\n"
                        f"  backup:  {manifest.get('jobs_columns')}\n"
                        f"  current: {list(Job.COLUMNS.keys())}"
                    )
                if manifest.get('files_columns') != list(Project.FILES_COLUMNS.keys()):
                    raise ValueError(
                        "Backup 'files' columns do not match current Project.FILES_COLUMNS:\n"
                        f"  backup:  {manifest.get('files_columns')}\n"
                        f"  current: {list(Project.FILES_COLUMNS.keys())}"
                    )

                # 2) Restore jobs table
                print_rank_0(f"Restoring table {self.project_name}.{Job.TABLE_NAME}...")
                jobs_data = tar.extractfile(Project.BACKUP_JOBS_NAME).read()
                copy_jobs_q = sql.SQL("COPY {} FROM STDIN WITH BINARY").format(
                    self.get_jobs_table_sql_identifier(),
                )
                cur = conn.cursor()
                cur.copy_expert(copy_jobs_q.as_string(conn), io.BytesIO(jobs_data))

                # 3) Restore files table
                print_rank_0(f"Restoring table {self.project_name}.{Project.FILES_TABLE_NAME}...")
                files_data = tar.extractfile(Project.BACKUP_FILES_NAME).read()
                copy_files_q = sql.SQL("COPY {} FROM STDIN WITH BINARY").format(
                    self.get_files_table_sql_identifier(),
                )
                cur = conn.cursor()
                cur.copy_expert(copy_files_q.as_string(conn), io.BytesIO(files_data))

                # 4) Restore on-disk binary files. We push them to the server
                #    via the write_bytes_to_file procedure so it works even if
                #    the local user has no direct write access to the folder.
                write_q = sql.SQL("SELECT {}.write_bytes_to_file(%s, %s)").format(
                    sql.Identifier(self.project_name),
                )

                results_count = 0
                checkpoints_count = 0
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    if member.name.startswith(Project.BACKUP_RESULTS_PREFIX):
                        rel = member.name[len(Project.BACKUP_RESULTS_PREFIX):]
                        if not rel:
                            continue
                        dst = os.path.join(self.res_prefix, rel)
                        results_count += 1
                    elif member.name.startswith(Project.BACKUP_CHECKPOINTS_PREFIX):
                        rel = member.name[len(Project.BACKUP_CHECKPOINTS_PREFIX):]
                        if not rel:
                            continue
                        dst = os.path.join(self.ckpnt_prefix, rel)
                        checkpoints_count += 1
                    else:
                        continue
                    data = tar.extractfile(member).read()
                    cur = conn.cursor()
                    cur.execute(write_q, (dst, data))

                print_rank_0(
                    f"Restored {results_count} result file(s) and "
                    f"{checkpoints_count} checkpoint file(s) on the database server."
                )

            if commit:
                conn.commit()

        return self.project_name

    def load_from_file(self, file_path:str, recreate:bool=True):
        """
        Restore project state from an archive previously produced by
        :meth:`save_to_file` into this project's schema (``self.project_name``).

        The destination schema name does not need to match the original one
        stored in the archive - this method effectively acts as a backup
        restore that can also "rename" a project (e.g. instantiate
        ``Project('project_example_05_copy')`` and load a backup of
        ``project_example_05`` into it).

        Args:
            file_path: Path to a ``.tar.gz`` archive produced by
                :meth:`save_to_file`.
            recreate: If True (default), the project's schema and folders are
                dropped and recreated cleanly before loading the backup data.
                If False, the project is assumed to already be empty and
                compatible with the backup.

        Returns:
            The name of the project that was restored into (only on rank 0).
        """
        return self._load_from_file_query(
            file_path=file_path,
            recreate=recreate,
        )

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
            query = sql.SQL("SELECT {}.check_python_user()")
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

            print(f"Listing permissions for traversal to {self.project_path}...")
            print(os.system(f'namei -l {self.project_path}'))

    def create(self, if_exists: str = 'skip'):
        self._create_query(if_exists=if_exists)
        self._set_get_result_methods_by_permissions()

    @only_rank_0
    def _set_get_result_methods_by_permissions(self):
        try:
            file_list = os.listdir(self.project_path)
        except:
            user = os.getenv('USER')
            print(f"Failed to list files in {self.project_path}, probably no permissions were granted for user {user}!")
            file_list = []
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
        remaining_time:float,
        elapsed_time:float,
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
            remaining_time:float, 
            elapsed_time:float,
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

    @only_rank_0
    @debounced_query
    def _get_non_scheduled_jobs_query(
        self,
        conn:psycopg2.extensions.connection=None,
    ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("SELECT * FROM {} WHERE status != 'SCHEDULED'")
            query = query.format(self.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query)
            jobs_rows = cur.fetchall()
        return jobs_rows

    def get_non_scheduled_jobs(self):
        jobs_rows = self._get_non_scheduled_jobs_query()
        jobs_rows = MPI.COMM_WORLD.bcast(jobs_rows, root=0)
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
        
    def construct_filter_code(self, filter_model, col):
        number_type_map = {
            "equals": ("WHERE {} = {}", lambda x: x),
            "notEqual": ("WHERE {} != {}", lambda x: x),
            "greaterThan": ("WHERE {} > {}", lambda x: x),
            'greaterThanOrEqual': ("WHERE {} >= {}", lambda x: x),
            "lessThan": ("WHERE {} < {}", lambda x: x),
            "lessThanOrEqual": ("WHERE {} <= {}", lambda x: x),
            "inRange": ("WHERE {} BETWEEN {} AND {}", lambda x: x),
            "blank": ("WHERE {} IS NULL", lambda x: x),
            "notBlank": ("WHERE {} IS NOT NULL", lambda x: x),
        }
        text_type_map = {
            "equals": ("WHERE {} = {}", lambda x: x),   
            "notEqual": ("WHERE {} != {}", lambda x: x),
            "contains": ("WHERE {} ILIKE {}", lambda x: f"%{x}%"),
            "notContains": ("WHERE {} NOT ILIKE {}", lambda x: f"%{x}%"),
            "startsWith": ("WHERE {} LIKE {}", lambda x: f"{x}%"),
            "endsWith": ("WHERE {} LIKE {}", lambda x: f"%{x}"),
            "blank": ("WHERE {} IS NULL", lambda x: x),
            "notBlank": ("WHERE {} IS NOT NULL", lambda x: x),
        }
        date_type_map = {
            "equals": ("WHERE {} = {}", lambda f, t: f),
            "notEqual": ("WHERE {} != {}", lambda f, t: f),
            "greaterThan": ("WHERE {} > {}", lambda f, t: f),
            "lessThan": ("WHERE {} < {}", lambda f, t: f),
            "inRange": ("WHERE {} BETWEEN {} AND {}", lambda f, t: (f, t)),
            "blank": ("WHERE {} IS NULL", lambda f, t: f),
            "notBlank": ("WHERE {} IS NOT NULL", lambda f, t: f),
        }
        if filter_model['filterType'] == 'number':
            if filter_model.get('operator') is not None:
                sql_filters = []
                for i in range(5):
                    condition = filter_model.get(f'condition{i}')
                    if condition is not None:
                        sql_template, value_formatter = number_type_map[condition['type']]
                        sql_q = sql.SQL(sql_template)
                        filter_value = value_formatter(condition['filter'])
                        sql_filters.append(sql_q.format(sql.Identifier(col), sql.Literal(filter_value)))
                return sql.SQL(filter_model.get('operator')).join(sql_filters)
            else:
                sql_template, value_formatter = number_type_map[filter_model['type']]
                sql_q = sql.SQL(sql_template)
                filter_value = value_formatter(filter_model['filter'])
                return sql_q.format(sql.Identifier(col), sql.Literal(filter_value))
        elif filter_model['filterType'] == 'text':
            if filter_model.get('operator') is not None:
                sql_filters = []
                for i in range(5):
                    condition = filter_model.get(f'condition{i}')
                    if condition is not None:
                        sql_template, value_formatter = text_type_map[condition['type']]
                        sql_q = sql.SQL(sql_template)
                        filter_value = value_formatter(condition['filter'])
                        sql_filters.append(sql_q.format(sql.Identifier(col), sql.Literal(filter_value)))
                return sql.SQL(filter_model.get('operator')).join(sql_filters)
            else:
                sql_template, value_formatter = text_type_map[filter_model['type']]
                sql_q = sql.SQL(sql_template)
                filter_value = value_formatter(filter_model['filter'])
                return sql_q.format(sql.Identifier(col), sql.Literal(filter_value))
        elif filter_model['filterType'] == 'date':
            if filter_model.get('operator') is not None:
                sql_filters = []
                for i in range(5):
                    condition = filter_model.get(f'condition{i}')
                    if condition is not None:
                        sql_template, value_formatter = date_type_map[condition['type']]
                        dateFrom = condition.get('dateFrom')
                        dateTo = condition.get('dateTo')
                        sql_q = sql.SQL(sql_template)
                        filter_value = value_formatter(dateFrom, dateTo)
                        sql_filters.append(sql_q.format(sql.Identifier(col), sql.Literal(filter_value)))
                return sql.SQL(filter_model.get('operator')).join(sql_filters)
            else:
                sql_template, value_formatter = date_type_map[filter_model['type']]
                sql_q = sql.SQL(sql_template)
                dateFrom = filter_model.get('dateFrom')
                dateTo = filter_model.get('dateTo')
                filter_value = value_formatter(dateFrom, dateTo)
                return sql_q.format(sql.Identifier(col), sql.Literal(filter_value))
        else:
            return None

    def construct_sort_code(self, sort_model):
        if sort_model['sort'] == 'asc':
            return sql.SQL("ORDER BY {} ASC").format(sql.Identifier(sort_model['colId']))
        elif sort_model['sort'] == 'desc':
            return sql.SQL("ORDER BY {} DESC").format(sql.Identifier(sort_model['colId']))
        else:
            return None
            
    @only_rank_0
    @debounced_query
    def _get_ag_grid_row_request_query(
        self,
        columns:list[str],
        getRowsRequest:dict,
        conn:psycopg2.extensions.connection=None,
    ):  
        with DBConnection() if conn is None else conn as conn:

            # WHERE clauses
            sql_filters = []
            for column, filter_model in getRowsRequest['filterModel'].items():
                sql_filters.append(self.construct_filter_code(filter_model, column))
            
            # ORDER BY clauses (add priority when nothing specified)
            if not getRowsRequest['sortModel']:
                getRowsRequest['sortModel'] = [{"colId": "priority", "sort": "asc"}]
            sql_sorts = []
            for sort_model in getRowsRequest['sortModel']:
                sql_sorts.append(self.construct_sort_code(sort_model))

            cte_query = sql.SQL("""
                WITH filtered_data AS (
                    SELECT {columns} FROM {table} {where}
                    {sort}
                )
                SELECT 
                    (SELECT COUNT(*) FROM filtered_data) AS total_count,
                    f.*
                FROM filtered_data f
                {limit}
            """).format(
                columns=sql.SQL(',').join([sql.Identifier(c) for c in columns]),
                table=self.get_jobs_table_sql_identifier(),
                where=sql.SQL(" ").join(sql_filters),
                sort=sql.SQL(" ").join(sql_sorts),
                limit=sql.SQL("LIMIT {} OFFSET {}").format(
                    sql.Literal(getRowsRequest['endRow'] - getRowsRequest['startRow']),
                    sql.Literal(getRowsRequest['startRow']),
                )
            )
            
            cur = conn.cursor()
            cur.execute(cte_query)
            result = cur.fetchall()
        if result:
            total_count = result[0][0]
            jobs_rows = [row[1:] for row in result]
            df = pd.DataFrame(jobs_rows, columns=columns)
            return df, total_count
        else:
            return pd.DataFrame(columns=columns), 0

    def get_jobs_request(
        self,
        columns:list[str],
        start_row:int,
        end_row:int,
    ):
        res = self._get_jobs_request_query(columns=columns, start_row=start_row, end_row=end_row)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    @debounced_query
    def _set_jsonb_column_query(
        self,
        signature:str,
        column:str,
        value:dict,
        conn:psycopg2.extensions.connection=None,
    ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("UPDATE {} SET {} = %s WHERE signature = %s")
            query = query.format(self.get_jobs_table_sql_identifier(), sql.Identifier(column))
            cur = conn.cursor()
            cur.execute(query, (json.dumps(value), signature))
            conn.commit()

    @only_rank_0
    @debounced_query
    def _append_jsonb_column_query(
        self,
        signature:str,
        column:str,
        value:dict,
        conn:psycopg2.extensions.connection=None,
    ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("UPDATE {} SET {} = {} || %s WHERE signature = %s")
            query = query.format(self.get_jobs_table_sql_identifier(), sql.Identifier(column))
            cur = conn.cursor()
            cur.execute(query, (json.dumps(value), signature))
            conn.commit()

    @only_rank_0
    @debounced_query
    def _get_jsonb_column_query(
        self,
        signature:str,
        column:str,
        conn:psycopg2.extensions.connection=None,
    ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("SELECT {} FROM {} WHERE signature = %s")
            query = query.format(sql.Identifier(column), self.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (signature, ))
            res = cur.fetchone()[0]
        return res

    def _set_output_build_query(
        self,
        signature:str,
        output:dict,
        conn:psycopg2.extensions.connection=None,
    ):
        self._set_jsonb_column_query(
            signature=signature,
            column='output_build',
            value=output,
            conn=conn,
        )

    def set_output_build(self, signature:str, output:dict):
        self._set_output_build_query(
            signature=signature,
            output=output,
        )

    def _set_postprocess_output_query(
        self,
        signature:str,
        output:dict,
        conn:psycopg2.extensions.connection=None,
    ):
        self._set_jsonb_column_query(
            signature=signature,
            column='output_postprocess',
            value=output,
            conn=conn,
        )

    def set_postprocess_output(self, signature:str, output:dict):
        self._set_postprocess_output_query(
            signature=signature,
            output=output,
        )

    @only_rank_0
    @debounced_query
    def _append_output_build_query(
        self,
        signature:str,
        output:dict,
        conn:psycopg2.extensions.connection=None,
    ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("UPDATE {} SET output_build = output_build || %s WHERE signature = %s")
            query = query.format(self.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (json.dumps(output), signature))
            conn.commit()

    def append_output_build(self, signature:str, output:dict):
        self._append_output_build_query(
            signature=signature,
            output=output,
        )

    @only_rank_0
    @debounced_query
    def _get_input_outputs_query(
        self, 
        signature:str,
        conn:psycopg2.extensions.connection=None,
    ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL("SELECT p_inputs, output_build, output_postprocess, error_log FROM {} WHERE signature = %s")
            query = query.format(self.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (signature, ))
            res = cur.fetchone()
            return {'inputs': res[0], 'output_build': res[1], 'output_postprocess': res[2], 'error_log': res[3]}

    def get_input_outputs(self, signature:str):
        res = self._get_input_outputs_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)

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
        priority:int|None=None,
        conn:psycopg2.extensions.connection=None,
        ):
        with DBConnection() if conn is None else conn as conn:
            base_query = (
                "SELECT * FROM {} WHERE ("
                "status = 'SCHEDULED' OR "
                "status LIKE 'FAILED%%' OR "
                "status = 'INTERRUPTED'"
                ")"
            )
            params: tuple = ()
            if priority is not None:
                base_query += " AND priority = %s"
                params = (priority,)
            base_query += " ORDER BY priority ASC LIMIT 1"

            query = sql.SQL(base_query).format(
                self.get_jobs_table_sql_identifier(),
                )
            cur = conn.cursor()
            cur.execute(query, params)
            row = cur.fetchone()

            if row is None:
                return None

            required_source_files = row[Job.COLUMNS_MAP['required_source_files']]
            source_files_rows = self._get_files_query(
                conn=conn,
                signatures=required_source_files,
            )
        return Job(dict(zip(Job.COLUMNS.keys(), row)), source_files_rows, self)

    def get_next_scheduled_job(self, priority:int|None=None) -> Job | None:
        res = self._get_next_scheduled_job_query(priority=priority)
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
    def _truncate_result_binary_file_query(
        self,
        signature: str,
        keep_n_bytes: int,
        conn: psycopg2.extensions.connection = None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("SELECT {}.truncate_file(%s, %s)")
            query = query.format(sql.Identifier(self.project_name))
            binary_file_path = os.path.join(
                self.res_prefix, 
                f'{signature}.hbres'
            )
            cur = conn.cursor()
            cur.execute(query, (binary_file_path, keep_n_bytes))
            if commit:
                conn.commit()

    def truncate_result_binary_file(self, signature: str, keep_n_bytes: int):
        """Truncates the result binary file at the specified position.
        
        Args:
            signature: The signature of the job whose result file should be truncated
            position: The position in bytes where the file should be truncated
                     (all data after this position will be removed)
        """
        self._truncate_result_binary_file_query(signature=signature, keep_n_bytes=keep_n_bytes)


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
    def _parse_binary_result_header(self, binary_data:bytes):
        size = len(binary_data)
        mem_view_bytes = memoryview(binary_data)
        n_cols = np.frombuffer(mem_view_bytes[:8], dtype=np.int64)[0]
        n_cols_names = np.frombuffer(mem_view_bytes[8:16], dtype=np.int64)[0]
        col_names = mem_view_bytes[16:16+n_cols_names].tobytes().decode('utf-8').split('\n')
        assert ((size-16-n_cols_names) % (n_cols * 8) == 0), (
            "Size of binary data is not divisible by number size of float64 (8 bytes)"
        )
        n_rows = int((size-16-n_cols_names) / (n_cols * 8))
        return n_cols, n_cols_names, col_names, n_rows

    @only_rank_0
    def _parse_binary_result_bytes(self, binary_data:bytes):
        n_cols, n_cols_names, col_names, n_rows = self._parse_binary_result_header(binary_data)
        start_time = time.time()
        mem_view_bytes = memoryview(binary_data)
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