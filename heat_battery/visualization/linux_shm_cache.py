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

DEFAULT_EXPIRATION_TIME = 600  # seconds


class DoubleBufferedSHMCache:
    """
    A cache that uses double-buffering to allow safe concurrent reads and writes.
    
    Strategy:
    - Each cache entry has two possible version files: .v0.data and .v1.data
    - A symlink (.current) points to the active version
    - Writing creates the NEW version, then atomically switches the symlink
    - Reading always goes through the symlink, so it always sees consistent data
    - After switching, the old version is deleted
    
    This eliminates race conditions where a reader could access a file being
    overwritten, which caused Bus Errors with memory-mapped files.
    """
    
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
        self.threadsafe = True  # This implementation is thread-safe by design
    
    def _get_base_path(self, key_route: tuple[str]) -> Path:
        """Get base path for a cache key (without version suffix)"""
        return self.cache_dir / "-".join(str(x) for x in key_route)
    
    def _get_versioned_path(self, key_route: tuple[str], version: int) -> Path:
        """Get path for a specific version"""
        return self._get_base_path(key_route).with_suffix(f'.v{version}.data')
    
    def _get_link_path(self, key_route: tuple[str]) -> Path:
        """Get path to the symlink that points to current version"""
        return self._get_base_path(key_route).with_suffix('.current')
    
    def _get_current_version(self, key_route: tuple[str]) -> Optional[int]:
        """Get the current active version number"""
        link = self._get_link_path(key_route)
        if not link.is_symlink():
            return None
        try:
            target = os.readlink(link)
            # Extract version from filename like "key.v0.data" -> 0
            # The target is just the filename (relative symlink)
            suffix = Path(target).suffixes  # e.g., ['.v0', '.data']
            if len(suffix) >= 2:
                version_str = suffix[0].replace('.v', '')
                return int(version_str)
        except (ValueError, AttributeError, OSError):
            pass
        return None
    
    def _entry_exists(self, key_route: tuple[str]) -> bool:
        """Check if a valid entry exists"""
        link = self._get_link_path(key_route)
        # Check if symlink exists AND points to a valid file
        return link.is_symlink() and link.exists()
    
    # Legacy alias
    def _get_path(self, key_route: tuple[str]) -> Path:
        """Legacy method - returns the symlink path for reading"""
        return self._get_link_path(key_route)
    
    def _atomic_write(self, key_route: tuple[str], write_func) -> None:
        """
        Atomically write data using double-buffering.
        
        Args:
            key_route: The cache key
            write_func: Function that takes a file path and writes data to it
        """
        current_version = self._get_current_version(key_route)
        # Toggle between version 0 and 1
        new_version = 0 if current_version is None else (1 - current_version)
        
        new_path = self._get_versioned_path(key_route, new_version)
        link_path = self._get_link_path(key_route)
        
        # Step 1: Write to new version file
        write_func(new_path)
        
        # Step 2: Atomically switch symlink
        # Create temp symlink, then atomically replace
        tmp_link = link_path.with_suffix('.tmp')
        try:
            # Remove old tmp link if exists
            tmp_link.unlink(missing_ok=True)
            # Create new symlink pointing to new version (relative path for portability)
            os.symlink(new_path.name, tmp_link)
            # Atomically replace the current symlink (this is atomic on Linux)
            os.replace(tmp_link, link_path)
        except Exception as e:
            # Cleanup on failure
            tmp_link.unlink(missing_ok=True)
            new_path.unlink(missing_ok=True)
            raise e
        
        # Step 3: Delete old version (safe now - no readers are using it)
        if current_version is not None:
            old_path = self._get_versioned_path(key_route, current_version)
            old_path.unlink(missing_ok=True)
    
    def write_bytes(
            self, 
            key_route: tuple[str], 
            data: bytes, 
            expiration_time: int = DEFAULT_EXPIRATION_TIME
        ) -> None:
        """Write raw bytes with double-buffering"""
        def write_func(path):
            with open(path, 'wb') as f:
                f.write(expiration_time.to_bytes(self.EXPIRATION_TIME_SIZE, 'little'))
                f.write(data)
        self._atomic_write(key_route, write_func)
    
    def read_bytes(self, key_route: tuple[str]) -> Optional[bytes]:
        """Read raw bytes (always reads through symlink for consistency)"""
        if not self._entry_exists(key_route):
            return None
        link_path = self._get_link_path(key_route)
        with open(link_path, 'rb') as f:
            f.seek(self.EXPIRATION_TIME_SIZE)
            return f.read()
    
    def append_bytes(
            self, 
            key_route: tuple[str], 
            data: bytes, 
            expiration_time: int | None = None
        ) -> None:
        """
        Append raw bytes to existing entry.
        Note: This reads current data, appends, and writes atomically.
        """
        current_data = self.read_bytes(key_route) or b''
        new_data = current_data + data
        exp_time = expiration_time if expiration_time is not None else DEFAULT_EXPIRATION_TIME
        self.write_bytes(key_route, new_data, exp_time)

    def write_small_dataframe(
            self, 
            key_route: tuple[str], 
            df: pd.DataFrame, 
            expiration_time: int = DEFAULT_EXPIRATION_TIME
        ) -> None:
        """Write dataframe with double-buffering"""
        def write_func(path):
            with open(path, 'wb') as f:
                f.write(expiration_time.to_bytes(self.EXPIRATION_TIME_SIZE, 'little'))
                f.write(df.to_parquet())
        self._atomic_write(key_route, write_func)

    def read_small_dataframe(self, key_route: tuple[str]) -> Optional[pd.DataFrame]:
        """Read dataframe with double-buffering"""
        if not self._entry_exists(key_route):
            return None
        link_path = self._get_link_path(key_route)
        return pd.read_parquet(link_path)
    
    def write_hbres_dataframe(
            self, 
            key_route: tuple[str], 
            df: pd.DataFrame, 
            expiration_time: int = DEFAULT_EXPIRATION_TIME
        ) -> None:
        """Write dataframe with double-buffering"""
        def write_func(path):
            columns = df.columns
            col_names_bytes = bytes('\n'.join(columns), 'utf-8')
            n_cols_bytes = np.int64(len(columns)).tobytes()
            n_cols_names_bytes = np.int64(len(col_names_bytes)).tobytes()
            data_bytes = df.values.astype(np.float64).tobytes()
            header_bytes = n_cols_bytes + n_cols_names_bytes + col_names_bytes
            
            with open(path, 'wb') as f:
                f.write(expiration_time.to_bytes(self.EXPIRATION_TIME_SIZE, 'little'))
                f.write(header_bytes)
                f.write(data_bytes)
        
        self._atomic_write(key_route, write_func)
    
    def read_hbres_dataframe(self, key_route: tuple[str]) -> Optional[pd.DataFrame]:
        """
        Read dataframe (reads through symlink for consistency).
        
        Uses memmap for efficient reading, then copies data to ensure safety
        even if the underlying file is deleted after symlink switch.
        """
        if not self._entry_exists(key_route):
            return None
        
        link_path = self._get_link_path(key_route)
        # Resolve the symlink to get actual file path for memmap
        actual_path = link_path.resolve()
        
        with open(link_path, 'rb') as f:
            f.seek(self.EXPIRATION_TIME_SIZE)
            size_data = f.read(16)
            n_cols = np.frombuffer(size_data[:8], dtype=np.int64)[0]
            n_cols_names = np.frombuffer(size_data[8:16], dtype=np.int64)[0]
            col_names = f.read(n_cols_names).decode('utf-8').split('\n')
        
        # Use actual resolved path for memmap
        file_size = os.path.getsize(actual_path)
        offset = self.EXPIRATION_TIME_SIZE + 16 + n_cols_names
        n_rows = int((file_size - offset) / (n_cols * 8))
        
        array = np.memmap(
            actual_path,
            dtype=np.float64,
            mode='r',
            offset=offset,
            shape=(n_rows, n_cols)
        )
        
        # IMPORTANT: Copy the data so we're not affected if the file is deleted
        # after symlink switch. The memmap file could be deleted after we return.
        df = pd.DataFrame(np.array(array), columns=col_names)
        return df
    
    def write_json(
            self, 
            key_route: tuple[str], 
            data: dict, 
            expiration_time: int = DEFAULT_EXPIRATION_TIME
        ) -> None:
        """Write JSON with double-buffering"""
        def write_func(path):
            with open(path, 'w') as f:
                # Write expiration as binary prefix, then JSON
                pass
            with open(path, 'wb') as f:
                f.write(expiration_time.to_bytes(self.EXPIRATION_TIME_SIZE, 'little'))
                f.write(json.dumps(data).encode('utf-8'))
        self._atomic_write(key_route, write_func)
    
    def read_json(self, key_route: tuple[str]) -> Optional[dict]:
        """Read JSON"""
        if not self._entry_exists(key_route):
            return None
        link_path = self._get_link_path(key_route)
        with open(link_path, 'rb') as f:
            f.seek(self.EXPIRATION_TIME_SIZE)
            return json.loads(f.read().decode('utf-8'))
    
    def write_object(
            self, 
            key_route: tuple[str], 
            obj: Any, 
            expiration_time: int = DEFAULT_EXPIRATION_TIME
        ) -> None:
        """Write pickled object with double-buffering"""
        def write_func(path):
            with open(path, 'wb') as f:
                f.write(expiration_time.to_bytes(self.EXPIRATION_TIME_SIZE, 'little'))
                cloudpickle.dump(obj, f)
        self._atomic_write(key_route, write_func)
    
    def read_object(self, key_route: tuple[str]) -> Optional[Any]:
        """Read pickled object"""
        if not self._entry_exists(key_route):
            return None
        link_path = self._get_link_path(key_route)
        with open(link_path, 'rb') as f:
            f.seek(self.EXPIRATION_TIME_SIZE)
            return cloudpickle.load(f)
    
    def delete(self, key_route: tuple[str]) -> None:
        """Delete a cache entry and all its versions"""
        freed_size = 0
        
        # Delete both version files
        for version in [0, 1]:
            path = self._get_versioned_path(key_route, version)
            if path.exists():
                freed_size += path.stat().st_size
                path.unlink(missing_ok=True)
        
        # Delete symlink
        link_path = self._get_link_path(key_route)
        if link_path.is_symlink():
            link_path.unlink(missing_ok=True)
        
        # Delete any tmp link
        tmp_link = link_path.with_suffix('.tmp')
        tmp_link.unlink(missing_ok=True)
        
        print(f"Freed {freed_size} bytes from cache {self.cache_name}")
    
    def clear_all(self) -> None:
        """Delete all cache entries"""
        freed_size = 0
        for path in self.cache_dir.iterdir():
            try:
                if path.is_symlink() or path.is_file():
                    stat = path.stat() if not path.is_symlink() else None
                    if stat:
                        freed_size += stat.st_size
                    path.unlink(missing_ok=True)
            except Exception:
                pass
        print(f"Freed {freed_size} bytes from cache {self.cache_name}")
        try:
            self.cache_dir.rmdir()
        except OSError:
            pass  # Directory not empty or other issue
    
    def clear_expired(self) -> None:
        """Delete expired cache entries"""
        freed_size = 0
        # Find all .current symlinks (each represents one cache entry)
        for link_path in self.cache_dir.glob('*.current'):
            if not link_path.is_symlink():
                continue
            try:
                # Read expiration time from the actual file
                with open(link_path, 'rb') as f:
                    exp_bytes = f.read(self.EXPIRATION_TIME_SIZE)
                    expiration_time = int.from_bytes(exp_bytes, 'little')
                
                stat = link_path.stat()
                if (stat.st_atime + expiration_time) < (time.time() + self.EXPIRATION_TIME_TOL):
                    # Entry is expired - delete it
                    # Extract key_route from link path
                    base_name = link_path.stem  # removes .current
                    # Delete version files
                    for version in [0, 1]:
                        v_path = link_path.with_suffix(f'.v{version}.data')
                        if v_path.exists():
                            freed_size += v_path.stat().st_size
                            v_path.unlink(missing_ok=True)
                    link_path.unlink(missing_ok=True)
            except Exception:
                pass
        print(f"Freed {freed_size} bytes from cache {self.cache_name}")
    
    def __del__(self):
        """Cleanup on deletion - commented out to preserve cache across restarts"""
        # Uncomment if you want cache cleared when object is destroyed:
        # try:
        #     self.clear_all()
        # except:
        #     pass
        pass
    
    def __sizeof__(self) -> int:
        """Size of cache in bytes"""
        total = 0
        for path in self.cache_dir.iterdir():
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except Exception:
                pass
        return total
    
    def __repr__(self) -> str:
        """String representation"""
        return f"DoubleBufferedSHMCache(name={self.cache_name}, size={self.__sizeof__()}, threadsafe={self.threadsafe})"


# Backwards compatibility alias
SHMCache = DoubleBufferedSHMCache


class ThreadSafeSHMCache(DoubleBufferedSHMCache):
    """
    Thread-safe version with additional file locking.
    
    Note: DoubleBufferedSHMCache is already thread-safe for most operations
    due to atomic symlink switching. This class adds extra locking for
    operations that might need stronger guarantees.
    """
    
    def __init__(self, cache_name: str):
        super().__init__(cache_name)
        self.lock_file = self.cache_dir / ".lock"
        self.lock = FileLock(self.lock_file)
    
    @contextlib.contextmanager
    def _key_lock(self, key_route: tuple[str]):
        """Lock for specific key operations"""
        lock_file = self._get_base_path(key_route).with_suffix('.lock')
        lock = FileLock(lock_file)
        with lock:
            yield
    
    def write_bytes(self, key_route: tuple[str], data: bytes, expiration_time: int = DEFAULT_EXPIRATION_TIME) -> None:
        with self._key_lock(key_route):
            super().write_bytes(key_route, data, expiration_time)
    
    def read_bytes(self, key_route: tuple[str]) -> Optional[bytes]:
        with self._key_lock(key_route):
            return super().read_bytes(key_route)
    
    def append_bytes(self, key_route: tuple[str], data: bytes, expiration_time: int = DEFAULT_EXPIRATION_TIME) -> None:
        with self._key_lock(key_route):
            super().append_bytes(key_route, data, expiration_time)
    
    def write_hbres_dataframe(self, key_route: tuple[str], df: pd.DataFrame, expiration_time: int = DEFAULT_EXPIRATION_TIME) -> None:
        with self._key_lock(key_route):
            super().write_hbres_dataframe(key_route, df, expiration_time)
    
    def read_hbres_dataframe(self, key_route: tuple[str]) -> Optional[pd.DataFrame]:
        with self._key_lock(key_route):
            return super().read_hbres_dataframe(key_route)
    
    def write_json(self, key_route: tuple[str], data: dict, expiration_time: int = DEFAULT_EXPIRATION_TIME) -> None:
        with self._key_lock(key_route):
            super().write_json(key_route, data, expiration_time)
    
    def read_json(self, key_route: tuple[str]) -> Optional[dict]:
        with self._key_lock(key_route):
            return super().read_json(key_route)
    
    def write_object(self, key_route: tuple[str], obj: Any, expiration_time: int = DEFAULT_EXPIRATION_TIME) -> None:
        with self._key_lock(key_route):
            super().write_object(key_route, obj, expiration_time)
    
    def read_object(self, key_route: tuple[str]) -> Optional[Any]:
        with self._key_lock(key_route):
            return super().read_object(key_route)
