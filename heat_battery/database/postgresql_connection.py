import psycopg2
import psycopg2.sql as sql
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool, SimpleConnectionPool
from ..config import (
    assert_config_feature_enabled,
    assert_config_value_set,
    get_config_item,
)

def assert_db_feature_enabled(feature: str):
    error_msg = (
        f"{feature} is disabled in the configuration file for safety reasons.\n"
        f"Please enable it in local configuration file 'config.yaml'."
    )
    assert_config_feature_enabled(['database', 'postgres', feature], error_msg)

def assert_db_config_value_set(value: str):
    error_msg = (
        f"{value.capitalize()} not set in the configuration file.\n"
        "Please set it in local configuration file 'config.yaml'."
    )
    assert_config_value_set(['database', 'postgres', value], error_msg)

def get_postgres_credentials():
    creds_entries = [
        f"user={get_config_item(['database', 'postgres', 'user'])}",
        f"password={get_config_item(['database', 'postgres', 'password'])}",
        f"host={get_config_item(['database', 'postgres', 'host'])}",
        f"port={get_config_item(['database', 'postgres', 'port'])}",
    ]
    creds = " ".join(creds_entries)
    return creds

def get_postgres_dsn(database_name: str):
    dsn = f"dbname={database_name} " + get_postgres_credentials()
    return dsn

def get_single_db_connection(database_name: str) -> psycopg2.extensions.connection:
    """This gives a conection to a single database within a postgres server"""
    return psycopg2.connect(get_postgres_dsn(database_name))

def get_postgres_version():
    conn = get_single_db_connection('postgres')
    cur = conn.cursor()
    cur.execute("SELECT VERSION()")
    version = cur.fetchone()[0]
    conn.close()
    return version

DB_VERSION = get_postgres_version()

DB_VERSION = get_postgres_version()

class SingleConnection:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.conn = psycopg2.connect(dsn)

    def check_alive_poll(self):
        try:
            self.conn.poll()
            return True
        except psycopg2.OperationalError:
            return False

    def check_alive_query(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone()[0] == 1
        except psycopg2.Error:
            return False

    def getconn(self):
        if not self.check_alive_poll():
            self.conn.close()
            self.conn = psycopg2.connect(self.dsn)
        return self.conn
    
    def putconn(self, conn):
        pass

if get_config_item(['database', 'postgres', 'connection_method']) == 'threaded_pool':
    DB_CONN_POOL = ThreadedConnectionPool(
        minconn=get_config_item(['database', 'postgres', 'min_pool_size']),
        maxconn=get_config_item(['database', 'postgres', 'max_pool_size']),
        dsn=get_postgres_dsn(get_config_item(['database', 'postgres', 'db_name']))
    )
elif get_config_item(['database', 'postgres', 'connection_method']) == 'simple_pool':
    DB_CONN_POOL = SimpleConnectionPool(
        minconn=get_config_item(['database', 'postgres', 'min_pool_size']),
        maxconn=get_config_item(['database', 'postgres', 'max_pool_size']),
        dsn=get_postgres_dsn(get_config_item(['database', 'postgres', 'db_name']))
    )
elif get_config_item(['database', 'postgres', 'connection_method']) == 'reuse_single':
    DB_CONN_POOL = SingleConnection(
        get_postgres_dsn(get_config_item(['database', 'postgres', 'db_name']))
    )
else:
    raise ValueError(
        f"Invalid value for 'connection_method' in configuration file. "
        f"Valid values are 'threaded_pool' and 'simple_pool'. "
        f"Got {get_config_item(['database', 'postgres', 'connection_method'])}."
    )

class DBConnection:
    def __init__(self):
        self.conn = DB_CONN_POOL.getconn()

    def __enter__(self):
        return self.conn
    
    def __exit__(self, exc_type, exc_value, traceback):
        DB_CONN_POOL.putconn(self.conn)
        #TODO: should we commit here?