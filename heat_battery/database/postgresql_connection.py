import time
import traceback
import psycopg2
import psycopg2.sql as sql
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool, SimpleConnectionPool
from ..config import (
    assert_config_feature_enabled,
    assert_config_value_set,
    get_config_item,
)

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

class DBConnection:
    DB_CONN_POOL = None

    def __init__(self):
        self.get_pool()
        self.conn = DBConnection.DB_CONN_POOL.getconn()

    def __enter__(self):
        return self.conn
    
    def __exit__(self, exc_type, exc_value, traceback):
        DBConnection.DB_CONN_POOL.putconn(self.conn)
        #TODO: should we commit here?

    def get_pool(self):
        """
        Initialize the connection pool if it is not already initialized so that 
        postgres credentials are not needed if there is no connection to the
        database.
        """
        if DBConnection.DB_CONN_POOL is None:
            if get_config_item(['database', 'postgres', 'connection_method']) == 'threaded_pool':
                DBConnection.DB_CONN_POOL = ThreadedConnectionPool(
                    minconn=get_config_item(['database', 'postgres', 'min_pool_size']),
                    maxconn=get_config_item(['database', 'postgres', 'max_pool_size']),
                    dsn=get_postgres_dsn(get_config_item(['database', 'postgres', 'db_name']))
                )
            elif get_config_item(['database', 'postgres', 'connection_method']) == 'simple_pool':
                DBConnection.DB_CONN_POOL = SimpleConnectionPool(
                    minconn=get_config_item(['database', 'postgres', 'min_pool_size']),
                    maxconn=get_config_item(['database', 'postgres', 'max_pool_size']),
                    dsn=get_postgres_dsn(get_config_item(['database', 'postgres', 'db_name']))
                )
            elif get_config_item(['database', 'postgres', 'connection_method']) == 'reuse_single':
                DBConnection.DB_CONN_POOL = SingleConnection(
                    get_postgres_dsn(get_config_item(['database', 'postgres', 'db_name']))
                )
            else:
                raise ValueError(
                    f"Invalid value for 'connection_method' in configuration file. "
                    f"Valid values are 'threaded_pool' and 'simple_pool'. "
                    f"Got {get_config_item(['database', 'postgres', 'connection_method'])}."
                )
            
def safe_query(func):
    #TODO: add loggin here per project
    """
    Retry a query a few times with debouncing if it fails due to a connection error.
    """
    def wrapper(*args, **kwargs):
        MAX_DELAY = 32
        MAX_ATTEMPTS = 5
        delay = 1
        attempts = 0
        while True:
            try:
                return func(*args, **kwargs)
            except psycopg2.Error as e:
                traceback.print_exc()
                time.sleep(delay)
                delay = min(2*delay, MAX_DELAY)
                attempts += 1
                if attempts > MAX_ATTEMPTS:
                    raise Exception(f"Too many attempts to perform: {func.__name__}!")
                continue
    return wrapper
