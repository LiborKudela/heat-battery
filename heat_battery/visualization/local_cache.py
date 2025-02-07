from pathlib import Path
import os
import time
from typing import Optional, Any
import struct
from filelock import FileLock
import contextlib
import pandas as pd
import json
import cloudpickle
import sys
import numpy as np
from io import BytesIO
DEFAULT_EXPIRATION_TIME = 600 # 1 week

class LocalCache:
    DATA_FILE_SUFFIX = '.data'
    EXPIRATION_TIME_SIZE = 8
    EXPIRATION_TIME_TOL = 1

    def __init__(self, cache_name: str):
        """
        Initialize cache directory in shared memory
        :param cache_name: Name of the cache
        """
        self.cache_name = cache_name
        self.data_store = {}
        self.threadsafe = False

    def _get_h_key(self, key_route: tuple[str]) -> str:
        """Get hash key for a cache key"""
        return hash(key_route)


    def write_pd_dataframe(
            self, 
            key_route: tuple[str], 
            df: pd.DataFrame, 
            expiration_time: int=DEFAULT_EXPIRATION_TIME, 
        ) -> None:
        """Write dataframe without compression"""
        h_key = self._get_h_key(key_route)
        self.data_store[h_key] = df

    def read_pd_dataframe(
            self, 
            key_route: tuple[str], 
        ) -> Optional[pd.DataFrame]:
        """Read dataframe"""
        start_time = time.time()
        h_key = self._get_h_key(key_route)
        df = self.data_store.get(h_key)
        print(f'Time taken to read dataframe: {time.time() - start_time:.5f}s')
        return df

    def write_json(
            self, 
            key_route: tuple[str], 
            data: dict, 
            expiration_time: int=DEFAULT_EXPIRATION_TIME, 
        ) -> None:
        """Write json"""
        h_key = self._get_h_key(key_route)
        self.data_store[h_key] = data

    def read_json(
            self, 
            key_route: tuple[str], 
        ) -> Optional[dict]:
        """Read json"""
        h_key = self._get_h_key(key_route)
        return self.data_store.get(h_key)

    def write_object(self, key_route: tuple[str], obj: Any, expiration_time: int=DEFAULT_EXPIRATION_TIME) -> None:
        """Write object"""
        h_key = self._get_h_key(key_route)
        self.data_store[h_key] = obj

    def read_object(self, key_route: tuple[str]) -> Optional[Any]:
        """Read object"""
        h_key = self._get_h_key(key_route)
        return self.data_store.get(h_key)

    def delete(self, key_route: tuple[str]) -> bool:        
        """Unlink specific cache entry"""
        h_key = self._get_h_key(key_route)
        del self.data_store[h_key]
        return True
    
    def clear_all(self) -> None:
        """Unlink all cache entries"""
        self.data_store.clear()

    def clear_expired(self) -> None:
        """Unlink expired cache entries"""
        pass

    def __del__(self):
        """Cleanup"""
        try:
            self.clear_all()
        except:
            pass

    def __sizeof__(self) -> int:
        """Size of cache"""
        return sum(sys.getsizeof(v) for v in self.data_store.values())
    
    def __repr__(self) -> str:
        """Representation"""
        return f"LocalCache(name={self.cache_name}, size={self.__sizeof__()}, threadsafe={self.threadsafe})"