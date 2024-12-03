import pandas as pd
from mpi4py import MPI
import datetime
import json
import os
from ..utilities import (
    hash_data,
    only_rank_0,
    print_rank_0,
)
from .jobs import Job

import psycopg2
import psycopg2.sql as sql
from psycopg2.extras import execute_values
from ..database.postgresql_connection import (
    DBConnection, 
    get_single_db_connection, 
    safe_query
)
from ..config import get_config_item

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

def create_database(if_exists:str='skip'):
    _create_database_query(if_exists=if_exists)

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
        self.project_name = project_name
        self.ensure_database_exists()
        self.create(if_exists=if_exists)

    def get_jobs_table_sql_identifier(self, project_name:str=None):
        project_name = project_name if project_name is not None else self.project_name
        return sql.Identifier(project_name, Job.TABLE_NAME)

    def get_files_table_sql_identifier(self, project_name:str=None):
        project_name = project_name if project_name is not None else self.project_name
        return sql.Identifier(project_name, Project.FILES_TABLE_NAME)

    def ensure_database_exists(self):
        _create_database_query(if_exists='skip')

    @only_rank_0
    @safe_query
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
    @safe_query
    def _drop_query(
        self,
        fail_if_not_exists: bool = False,
        cascade: bool = False,
        conn:psycopg2.extensions.connection = None, 
        ):

        commit = conn is None

        with DBConnection() if commit else conn as conn:

            query = sql.SQL("DROP SCHEMA {} {} {}")
            query = query.format(
                sql.SQL('IF EXISTS') if fail_if_not_exists else sql.SQL(''),
                sql.Identifier(self.project_name),
                sql.SQL('CASCADE') if cascade else sql.SQL(''),
            )
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()

    def drop(
        self,
        fail_if_not_exists: bool = False,
        cascade: bool = False,
    ):
        self._drop_query(fail_if_not_exists=fail_if_not_exists, cascade=cascade)

    @only_rank_0
    @safe_query
    def _create_query(
        self,
        conn:psycopg2.extensions.connection = None,
        if_exists: str = 'skip',
        ):

        commit = conn is None
        project_name = self.project_name

        with DBConnection() if commit else conn as conn:
            print(self._exists_query(conn))
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

            query = sql.SQL("CREATE SCHEMA IF NOT EXISTS {}")
            query = query.format(sql.Identifier(project_name))
            cur = conn.cursor()
            cur.execute(query)

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

            if commit:
                conn.commit()

    def create(self, if_exists: str = 'skip'):
        self._create_query(if_exists=if_exists)

    @only_rank_0
    @safe_query
    def _get_info_query(self, conn:psycopg2.extensions.connection=None):
        with DBConnection() if conn is None else conn as conn:

            query = sql.SQL(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s"
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
            print(query.as_string(conn))
            cur.execute(query)
            table_data = cur.fetchall()
            res = {t_name[0]: table_data[0] for t_name, table_data in zip(reversed(table_names), table_data)}
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
    @safe_query
    def _get_remote_node_name_query(
            self,
            signature:str, 
            conn:psycopg2.extensions.connection=None,
        ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("SELECT active_node_address FROM {} WHERE signature = %s")
            query = query.format(self.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (signature,))
            result = cur.fetchone()[0]
        return result 
    
    def get_remote_node_name(self, signature:str):
        res = self._get_remote_node_name_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    @safe_query
    def _set_remote_node_name_query(
            self,
            signature:str,
            active_node_address:str,
            conn:psycopg2.extensions.connection=None,
        ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("UPDATE {} SET active_node_address = %s WHERE signature = %s")
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
    @safe_query
    def _get_status_query(
        self,
        signature:str,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("SELECT status FROM {} WHERE signature = %s")
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
    @safe_query
    def _set_status_query(
        self, 
        signature:str,
        status:str,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("UPDATE {} SET status = %s, last_updated = NOW() WHERE signature = %s")
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
    @safe_query
    def _get_last_updated_query(self, signature:str, conn:psycopg2.extensions.connection=None):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("SELECT last_updated FROM {} WHERE signature = %s")
            query = query.format(self.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (signature,))
            result = cur.fetchone()[0]
        return result
    
    def get_last_updated(self, signature:str):
        res = self._get_last_updated_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    @safe_query
    def _get_error_log_query(
        self, 
        signature:str,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("SELECT error_log FROM {} WHERE signature = %s")
            query = query.format(self.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (signature,))
            result = cur.fetchone()[0]
        return result
    
    def get_error_log(self, signature:str):
        res = self._get_error_log_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    @safe_query
    def _set_error_log_query(
        self, 
        signature:str,
        error_log:str,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("UPDATE {} SET error_log = %s WHERE signature = %s")
            query = query.format(self.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (error_log, signature))
            if commit:
                conn.commit()

    def set_error_log(self, signature:str, error_log:str):
        self._set_error_log_query(signature=signature, error_log=error_log)

    @only_rank_0
    @safe_query
    def _update_progress_query(
        self, 
        signature:str,
        progress:float,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("UPDATE {} SET progress = %s, last_updated = NOW() WHERE signature = %s")
            query = query.format(self.get_jobs_table_sql_identifier())
            cur = conn.cursor()
            cur.execute(query, (progress, signature))
            if commit:
                conn.commit()

    def update_progress(self, signature:str, progress:float):
        self._update_progress_query(signature=signature, progress=progress)

    @only_rank_0
    @safe_query
    def _reset_all_statuses_query(
        self,   
        new_status:str='SCHEDULED',
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("UPDATE {} SET status = %s")
            query = query.format(
                self.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (new_status,))
            if commit:
                conn.commit()

    def reset_all_statuses(self, new_status:str='SCHEDULED'):
        self._reset_all_statuses_query(new_status=new_status)

    @only_rank_0
    @safe_query
    def _reset_uncompleted_jobs_status_query(
        self,
        new_status:str='INTERRUPTED',
        inactivity_minutes:int=5,
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL(
                "UPDATE {} SET "
                "status = %s, "
                "active_node_address = 'UNASSIGNED' "
                #"error_log = 'Cleared due to inactivity' "
                "WHERE (status LIKE 'RUNNING - %%' OR status = 'FAILED') " # says running
                "AND last_updated < NOW() - INTERVAL '%s minutes'" # but is not actualy running
            )
            query = query.format(
                self.get_jobs_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (new_status, inactivity_minutes))
            if commit:
                conn.commit()
 
    def reset_uncompleted_jobs_status(self, new_status:str='INTERRUPTED'):
        self._reset_uncompleted_jobs_status_query(new_status=new_status)

    @only_rank_0
    @safe_query
    def _add_files_query(
        self, 
        files_rows:list, 
        conn:psycopg2.extensions.connection=None,
        ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL(
                "INSERT INTO {} ({}) VALUES %s ON CONFLICT DO NOTHING"
            ).format(
                self.get_files_table_sql_identifier(),
                sql.SQL(', ').join(
                    sql.Identifier(col) for col in Project.FILES_COLUMNS.keys()
                ),
            )
            cur = conn.cursor()
            execute_values(cur, query, files_rows)

            if commit:
                conn.commit()

    def add_files(self, files_rows:list):
        self._add_files_query(files_rows=files_rows)

    @only_rank_0
    @safe_query
    def _check_files_exist_query(self, signatures:list[str], conn):
        with DBConnection() as conn:
            query = sql.SQL("SELECT EXISTS (SELECT 1 FROM {} WHERE signature IN %s)")
            query = query.format(
                self.get_files_table_sql_identifier()
            )
            cur = conn.cursor()
            cur.execute(query, (tuple(signatures), ))
            res = cur.fetchone()[0]
        return res

    def check_files_exist(self, signatures:list[str]):
        res = self._check_files_exist_query(signatures=signatures)
        return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    @safe_query
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
    @safe_query
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

                # set remote probe output location
                cp_data['p_inputs']['sim_p']['result_database'] = get_config_item(['database', 'postgres', 'db_name'])
                cp_data['p_inputs']['sim_p']['project_name'] = self.project_name
                cp_data['p_inputs']['sim_p']['database_table'] = f'res_{cp_data["signature"]}'

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
    @safe_query
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
                    files_rows.append((file_name, data, f'mesh_{ext}', hash_data(data)))
        with DBConnection() if commit else conn as conn:
            self._add_files_query(files_rows=files_rows, conn=conn)

            if commit:
                conn.commit()

    def add_meshes(self, dir:str, names:list[str]):
        self._add_meshes_query(dir=dir, names=names)

    @only_rank_0
    @safe_query
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
                    with open(os.path.join(dir, data_row[Project.FILES_COLUMNS_MAP['file_name']]), 'wb') as f:
                        f.write(data_row[Project.FILES_COLUMNS_MAP['data']])
    
    def get_meshes(self, names:list[str], dir:str):
        res = self._get_meshes_query(dir=dir, names=names)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    @safe_query
    def _check_meshes_exist_query(
        self, 
        names:list[str], 
        conn:psycopg2.extensions.connection=None,
    ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("SELECT EXISTS (SELECT 1 FROM {} WHERE file_name IN %s)")
            query = query.format(
                self.get_files_table_sql_identifier(),
            )
            cur = conn.cursor()
            cur.execute(query, (tuple(names), ))
            res = cur.fetchone()[0]
        return res

    def check_meshes_exist(self, names:list[str]):
        res = self._check_meshes_exist_query(names=names)
        return MPI.COMM_WORLD.bcast(res, root=0)
    
    @only_rank_0
    @safe_query
    def _get_jobs_query(
        self, 
        conn:psycopg2.extensions.connection=None, 
        as_dataframe:bool=False,
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
        jobs_rows = self._get_jobs_query(as_dataframe=as_dataframe)
        jobs_rows = MPI.COMM_WORLD.bcast(jobs_rows, root=0)
        if as_dataframe:
            return pd.DataFrame(jobs_rows, columns=Job.COLUMNS.keys())
        else:
            return [Job(dict(zip(Job.COLUMNS.keys(), row)), [], self) for row in jobs_rows]  

    @only_rank_0
    @safe_query
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
            source_files_rows = self._get_files_query(conn=conn, signatures=required_source_files)
        if row is not None:
            return Job(dict(zip(Job.COLUMNS.keys(), row)), source_files_rows, self)

    def get_next_scheduled_job(self) -> Job | None:
        res = self._get_next_scheduled_job_query()
        res = MPI.COMM_WORLD.bcast(res, root=0)
        MPI.COMM_WORLD.barrier()
        return res

    @only_rank_0
    @safe_query
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
            source_files_rows = self._get_files_query(conn=conn, signatures=required_source_files)
        if row is not None:
            return Job(dict(zip(Job.COLUMNS.keys(), row)), source_files_rows, self)
        
    def get_job(self, signature:str):
        res = self._get_job_query(signature=signature)
        return MPI.COMM_WORLD.bcast(res, root=0)

    # @only_rank_0
    # def _get_interupted_jobs_query(self, conn:psycopg2.extensions.connection=None):
    #     with DBConnection() if conn is None else conn as conn:
    #         query = sql.SQL("SELECT * FROM {} WHERE status != 'COMPLETED' AND last_updated < NOW() - INTERVAL '10 minutes'")
    #         query = query.format(
    #             self.get_jobs_table_sql_identifier(),
    #         )
    #         cur = conn.cursor()
    #         cur.execute(query)
    #         res_rows = cur.fetchall()
    #         return [Job(dict(zip(Job.COLUMNS.keys(), row)), [], self) for row in res_rows]   

    # def get_interupted_jobs(self):
    #     res = self._get_interupted_jobs_query()
    #     return MPI.COMM_WORLD.bcast(res, root=0)

    @only_rank_0
    @safe_query
    def _get_result_table_names_query(self):
        with DBConnection() as conn:
            query = sql.SQL("SELECT table_name FROM information_schema.tables WHERE table_schema = %s AND table_name LIKE 'res\_%%'")
            cur = conn.cursor()
            cur.execute(query, (self.project_name, ))
            res = cur.fetchall()
        return [row[0] for row in res]
    
    def get_result_table_names(self, as_dataframe:bool=False):
        res = self._get_result_table_names_query()
        res = MPI.COMM_WORLD.bcast(res, root=0)
        if as_dataframe:
            return pd.DataFrame(res, columns=['table_name'])
        else:
            return res
    
    @only_rank_0
    @safe_query
    def _get_result_table_query(
        self, 
        signature:str,
    ):
        with DBConnection() as conn:

            # check if table exists
            table_exists_query = sql.SQL("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s)")
            cur = conn.cursor()
            cur.execute(table_exists_query, (self.project_name, f'res_{signature}'))
            table_exists = cur.fetchone()[0]
            if not table_exists:
                print(f"Table 'res_{signature}' does not exist!")
                return None, None

            # get columns
            cols_query = sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s")
            cur = conn.cursor()
            cur.execute(cols_query, (self.project_name, f'res_{signature}'))
            cols = [col[0] for col in cur.fetchall()]

            # get rows
            query = sql.SQL("SELECT * FROM {} ORDER BY progress ASC")
            query = query.format(
                sql.Identifier(self.project_name, f'res_{signature}'),
            )
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
        return cols, rows
    
    def get_result_table(
            self, 
            signature:str, 
            as_dataframe:bool=False,
        ):
        res = self._get_result_table_query(signature=signature)
        res = MPI.COMM_WORLD.bcast(res, root=0)
        cols, rows = res
        if as_dataframe:
            return pd.DataFrame(columns=cols, data=rows)
        else:
            return res
        
    @only_rank_0
    @safe_query
    def _upload_result_table_query(
        self, 
        signature:str, 
        df:pd.DataFrame, 
        conn:psycopg2.extensions.connection=None,
        ):
        commit = conn is None
        with DBConnection() if commit else conn as conn:
            query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})")
            query = query.format(
                sql.Identifier(self.project_name, f'res_{signature}'),
                sql.SQL(', ').join(
                    sql.SQL("{} {}").format(
                        sql.Identifier(col),
                        sql.SQL("FLOAT")
                    ) for col in df.columns
                )
            )
            cur = conn.cursor()
            cur.execute(query)

            # insert data
            insert_query = sql.SQL("INSERT INTO {} ({}) VALUES %s")
            insert_query = insert_query.format(
                sql.Identifier(self.project_name, f'res_{signature}'),
                sql.SQL(', ').join(
                    sql.SQL("{}").format(
                        sql.Identifier(col)
                    ) for col in df.columns
                ),
            )
            execute_values(cur, insert_query, df.values, page_size=100000)

            if commit:
                conn.commit()

    def upload_result_table(
            self, 
            signature:str, 
            df:pd.DataFrame, 
        ):
        self._upload_result_table_query(signature=signature, df=df)

    @only_rank_0
    @safe_query
    def _clean_files_query(
        self, 
        conn:psycopg2.extensions.connection=None, 
        ):
        with DBConnection() if conn is None else conn as conn:
            query = sql.SQL(
                "SELECT array_agg(uq) FROM "
                "(SELECT DISTINCT unnest(required_source_files) as uq FROM {})"
                "subquery"
            )
            query = query.format(
                self.get_jobs_table_sql_identifier(self.project_name),
                )
            cur =  conn.cursor()
            cur.execute(query)
            res = cur.fetchone()[0]
        return res
    
    def clean_files(self):
        res = self._clean_files_query()
        return MPI.COMM_WORLD.bcast(res, root=0)
 
    def __repr__(self):
        return f"Project(name={self.project_name})"

@only_rank_0
@safe_query
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
