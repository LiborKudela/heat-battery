from pathlib import Path
import os
import time
from typing import Optional, Any
from filelock import FileLock
import contextlib
import pandas as pd
import json
import cloudpickle
import sys
import numpy as np

from heat_battery.utilities import hash_data
DEFAULT_EXPIRATION_TIME = 600 # 1 week

class SHMCache:
    DATA_FILE_SUFFIX = '.data'
    EXPIRATION_TIME_SIZE = 8
    EXPIRATION_TIME_TOL = 1

    def __init__(self, cache_name: str):
        """
        Initialize cache directory in shared memory
        :param cache_name: Name of the cache
        """
        self.shm_path = Path("/dev/shm")
        self.cache_name = cache_name
        self.cache_dir = self.shm_path / self.cache_name
        self.cache_dir.mkdir(exist_ok=True)
        self.threadsafe = False
    
    def _get_path(self, key_route: tuple[str]) -> Path:
        """Get path for a cache key"""
        path = self.cache_dir / "-".join(str(x) for x in key_route)
        print(path  )
        return path.with_suffix(SHMCache.DATA_FILE_SUFFIX)
    
    def _entry_exists(
        self, 
        key_route: tuple[str], 
        ) -> bool:
        """Check if file exists"""
        path = self._get_path(key_route)
        return path.exists()
    
    def write_bytes(
            self, 
            key_route: tuple[str], 
            data: bytes, 
            expiration_time: int=DEFAULT_EXPIRATION_TIME, 
        ) -> None:
        """Write raw bytes in selected mode and update expiration timestamp"""
        path = self._get_path(key_route)
        assert sys.getsizeof(expiration_time) == SHMCache.EXPIRATION_TIME_SIZE, "Expiration time must be 8 bytes (int64)"
        with open(path, 'wb') as f:
            f.write(expiration_time.to_bytes(SHMCache.EXPIRATION_TIME_SIZE, 'little'))
            f.write(data)

    def append_bytes(
            self, 
            key_route: tuple[str], 
            data: bytes, 
            expiration_time: int|None=None, 
        ) -> None:
        """Append raw bytes and change expiration time (optional)"""
        path = self._get_path(key_route)
        assert sys.getsizeof(expiration_time) == SHMCache.EXPIRATION_TIME_SIZE, "Expiration time must be 8 bytes (int64)"
        with open(path, 'rb+') as f:
            if expiration_time is not None:
                f.seek(0)
                f.write(expiration_time.to_bytes(SHMCache.EXPIRATION_TIME_SIZE, 'little'))
            f.seek(0, 2)
            f.write(data)
    
    def read_bytes(self, key_route: tuple[str]) -> Optional[bytes]:
        """Read raw bytes"""
        if not self._file_exists(key_route):
            return None
        with open(self._get_path(key_route), 'rb') as f:
            f.seek(SHMCache.EXPIRATION_TIME_SIZE)
            return f.read()

    def write_hbres_dataframe(
            self, 
            key_route: tuple[str], 
            df: pd.DataFrame, 
            expiration_time: int=DEFAULT_EXPIRATION_TIME, 
        ) -> None:
        """Write dataframe without compression"""
        columns = df.columns
        col_names_bytes = bytes('\n'.join(columns), 'utf-8')
        n_cols_bytes = np.int64(len(columns)).tobytes()
        n_cols_names_bytes = np.int64(len(col_names_bytes)).tobytes()
        data_bytes = df.values.tobytes()
        header_bytes = n_cols_bytes + n_cols_names_bytes + col_names_bytes
        path = self._get_path(key_route)
        with open(path, 'wb') as f:
            f.write(expiration_time.to_bytes(SHMCache.EXPIRATION_TIME_SIZE, 'little'))
            f.write(header_bytes)
            f.write(data_bytes)

    def read_hbres_dataframe(
            self, 
            key_route: tuple[str], 
        ) -> Optional[pd.DataFrame]:
        """Read dataframe"""
        if not self._entry_exists(key_route):
            return None
        path = self._get_path(key_route)
        with open(path, 'rb') as f:
            f.seek(SHMCache.EXPIRATION_TIME_SIZE)
            size_data = f.read(16)
            n_cols = np.frombuffer(size_data[:8], dtype=np.int64)[0]
            n_cols_names = np.frombuffer(size_data[8:16], dtype=np.int64)[0]
            col_names = f.read(n_cols_names).decode('utf-8').split('\n')
            array = np.memmap(
                path,
                dtype=np.float64,
                mode='r',
                offset=SHMCache.EXPIRATION_TIME_SIZE + 16 + n_cols_names,
                shape=(int((os.path.getsize(path) - SHMCache.EXPIRATION_TIME_SIZE - 16 - n_cols_names) / (n_cols * 8)), n_cols)
            )
            df = pd.DataFrame(array, columns=col_names, copy=False)
        return df

    def write_json(
            self, 
            key_route: tuple[str], 
            data: dict, 
            expiration_time: int=DEFAULT_EXPIRATION_TIME, 
        ) -> None:
        """Write json"""
        path = self._get_path(key_route)
        with open(path, 'wb') as f:
            f.write(expiration_time.to_bytes(SHMCache.EXPIRATION_TIME_SIZE, 'little'))
            json.dump(data, f)

    def read_json(
            self, 
            key_route: tuple[str], 
        ) -> Optional[dict]:
        """Read json"""
        if not self._entry_exists(key_route):
            return None
        path = self._get_path(key_route)
        with open(path, 'rb') as f:
            f.seek(SHMCache.EXPIRATION_TIME_SIZE)
            return json.load(f)

    def write_object(self, key_route: tuple[str], obj: Any, expiration_time: int=DEFAULT_EXPIRATION_TIME) -> None:
        """Write object"""
        path = self._get_path(key_route)
        with open(path, 'wb') as f:
            f.write(expiration_time.to_bytes(SHMCache.EXPIRATION_TIME_SIZE, 'little'))
            cloudpickle.dump(obj, f)

    def read_object(self, key_route: tuple[str]) -> Optional[Any]:
        """Read object"""
        path = self._get_path(key_route)
        if not path.exists():
            return None
        with open(path, 'rb') as f:
            f.seek(SHMCache.EXPIRATION_TIME_SIZE)
            return cloudpickle.load(f)

    def delete(self, key_route: tuple[str]) -> bool:
        """Unlink specific cache entry"""
        path = self._get_path(key_route)
        stat = path.stat()
        path.unlink(missing_ok=True)
        print(f"Freed {stat.st_size} bytes from cache {self.cache_name}")
    
    def clear_all(self) -> None:
        """Unlink all cache entries"""
        freed_size = 0
        for path in self.cache_dir.iterdir():
            stat = path.stat()
            path.unlink(missing_ok=True)
            freed_size += stat.st_size
        print(f"Freed {freed_size} bytes from cache {self.cache_name}")
        self.cache_dir.rmdir()

    def clear_expired(self) -> None:
        """Unlink expired cache entries"""
        freed_size = 0
        for path in self.cache_dir.iterdir():
            expiration_time = self.read_expiration_time(path)
            stat = path.stat()
            if (stat.st_atime + expiration_time) < (time.time() + SHMCache.EXPIRATION_TIME_TOL):
                path.unlink(missing_ok=True)
                freed_size += stat.st_size
        print(f"Freed {freed_size} bytes from cache {self.cache_name}")

    def __del__(self):
        """Cleanup"""
        try:
            self.clear()
        except:
            pass

    def __sizeof__(self) -> int:
        """Size of cache"""
        return sum(path.stat().st_size for path in self.cache_dir.iterdir())
    
    def __repr__(self) -> str:
        """Representation"""
        return f"SHMCache(name={self.cache_name}, size={self.__sizeof__()}, threadsafe={self.threadsafe})"

class ThreadSafeSHMCache(SHMCache):
    def __init__(self, cache_name: str):
        super().__init__(cache_name)
        self.lock_file = self.cache_dir / ".lock"
        self.lock = FileLock(self.lock_file)
        self.threadsafe = True

    @contextlib.contextmanager
    def _key_lock(self, key: str):
        """Lock for specific key operations"""
        lock_file = self._get_path(key).with_suffix('.lock')
        lock = FileLock(lock_file)
        with lock:
            yield
    
    def write_bytes(self, key_route: tuple[str], data: bytes, expiration_time: int=DEFAULT_EXPIRATION_TIME) -> None:
        with self._key_lock(key_route):
            super().write_bytes(key_route, data, expiration_time)
    
    def read_bytes(self, key_route: tuple[str]) -> Optional[bytes]:
        with self._key_lock(key_route):
            return super().read_bytes(key_route)
        
    def append_bytes(self, key_route: tuple[str], data: bytes, expiration_time: int=DEFAULT_EXPIRATION_TIME) -> None:
        with self._key_lock(key_route):
            super().append_bytes(key_route, data, expiration_time)

    def write_pd_dataframe(self, key_route: tuple[str], df: pd.DataFrame, expiration_time: int=DEFAULT_EXPIRATION_TIME) -> None:
        with self._key_lock(key_route):
            super().write_pd_dataframe(key_route, df, expiration_time)
    
    def read_pd_dataframe(self, key_route: tuple[str]) -> Optional[pd.DataFrame]:
        with self._key_lock(key_route):
            return super().read_pd_dataframe(key_route)

    def write_json(self, key_route: tuple[str], data: dict, expiration_time: int=DEFAULT_EXPIRATION_TIME) -> None:
        with self._key_lock(key_route):
            super().write_json(key_route, data, expiration_time)
    
    def read_json(self, key_route: tuple[str]) -> Optional[dict]:
        with self._key_lock(key_route):
            return super().read_json(key_route)
    
    def write_object(self, key_route: tuple[str], obj: Any, expiration_time: int=DEFAULT_EXPIRATION_TIME) -> None:
        with self._key_lock(key_route):
            super().write_object(key_route, obj, expiration_time)
    
    def read_object(self, key_route: tuple[str]) -> Optional[Any]:
        with self._key_lock(key_route):
            return super().read_object(key_route)