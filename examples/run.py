from mpi4py import MPI
import os

def print_r0(*args, **kwargs):
    # print only from rank zero when in MPI env.
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)

def run_single_example(dir, is_main=False):

    # local imports to avoid scope var collisions
    if not is_main:
        # if not runing as main script add relative import dot
        dir = '.' + dir
    exec(f'from {dir}.run_example import run')
    exec('run()')

def run_selected(dirs, is_main=False):

    print_r0(f"Selected examples {dirs}")
    
    success = []
    for d in dirs:
        try:
            print_r0(f"Running example: {d}")
            run_single_example(d, is_main=is_main)
            print_r0(f"{d} finished successfully:")
            success.append(True)
        except Exception as e:
            print_r0(f"Example {d} failed:")
            print_r0(e)
            success.append(False)
    return success

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", help="which example to run (defaults to all)")

    args = parser.parse_args()
    if not args.example:
        dirs = [
            'Example_01',
            'Example_02',
            'Example_03',
            'Example_04',
            ]
    else:

        dirs = args.example.split(',')   

    run_selected(dirs, is_main=True)