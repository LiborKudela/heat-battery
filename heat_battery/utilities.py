import cloudpickle
from mpi4py import MPI
import time
import datetime
import hashlib
import json
import os

def test_package():
    #TODO: make this proper
    return 0

def save_data(filepath, data, only_root=True):
    if not only_root or MPI.COMM_WORLD.rank == 0:
        with open(filepath, 'wb') as fp:
            cloudpickle.dump(data, fp)
    return None

def load_data(filepath, out=None):
    with open(filepath, 'rb') as fp:
        data = cloudpickle.load(fp)
    return data

class ProgressBar():
    def __init__(self, desc="", update_cb=None, n=20):
        self.desc = desc
        self.update_cb = update_cb
        self.percentage = 0.0
        self.prev_percentage = 0.0
        self.n = n
        self.fill_char = "█"
        self.unfill_char = "░"
        self.start_t = time.time()
        self.last_t = self.start_t

    def remain_time(self):
        iter_dt = time.time() - self.last_t
        iter_dp = self.percentage - self.prev_percentage
        return round((100 - self.percentage)/(iter_dp/iter_dt))

    def update(self):
        self.percentage = self.update_cb()
        t = time.time()

        l_bar = f"{self.desc}: {self.percentage:3.3f}%"

        filled_n = int(self.percentage/100*self.n)
        unfiled_n = self.n - filled_n
        bar = filled_n*self.fill_char+unfiled_n*self.unfill_char

        elapsed = round(t - self.start_t)
        remain = self.remain_time()
        if self.percentage < 100.0:
            e_str = str(datetime.timedelta(seconds=elapsed))
            r_str = str(datetime.timedelta(seconds=remain))
            r_bar = f"{e_str}>{r_str}       "
            print(f"{l_bar}|{bar}|{r_bar}", end="\r", flush=True)
        else:
            e_str = str(datetime.timedelta(seconds=elapsed))
            r_bar = f"Finished in {e_str}"
            print(f"{l_bar}|{bar}|{r_bar}", end="\n", flush=True)

        self.last_t = t
        self.prev_percentage = self.percentage

    def finish(self):
        pass

    def print_message(self, str):
        print(str, flush=True)

def hash_data(data):
    m = hashlib.sha256()
    d_str = json.dumps(data)
    m.update(d_str.encode('UTF-8'))
    return m.hexdigest()