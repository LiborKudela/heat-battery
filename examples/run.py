from mpi4py import MPI
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

def run_single_example(dir):

    # local imports to avoid scope var collisions
    exec(f'from {dir}.run_example import run')
    exec('run()')

def print_r0(*args, **kwargs):
    # print only from rank zero when in MPI env.
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)

def run_selected():

    print_r0(f"Selected examples {dirs}")
    
    for d in dirs:
        try:
            print_r0(f"Running example: {d}")
            run_single_example(d)
            print_r0(f"{d} finished successfully:")
        except Exception as e:
            print_r0(f"Example {d} failed:")
            print_r0(e)

if __name__ == '__main__':
    run_selected()