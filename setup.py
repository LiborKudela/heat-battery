from setuptools import setup
from setuptools import find_packages

# Load the Readme file.
with open(file="README.md", mode="r") as readme_handle:
    long_description = readme_handle.read()

setup(
    name = 'heat_battery',
    author = 'Libor Kudela',
    author_email = 'libor.kudela1@gmail.com',
    version = '0.0.1',
    description = 'FEM simulation/optimization of solid thermal storages',
    package_data={},
    include_package_data=True,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url='https://github.com/LiborKudela/heat-battery',
    install_requires=[
        'mpi4py==3.1.5',
        #'petsc4py>=3.15.1',
        'cloudpickle>=3.0.0',
        'pandas>=2.1.2',
        'plotly>=5.18.0',
        'plotly-resampler>=0.9.1',
        'numba>=0.58.1',
        'numpy==1.26.4',
        'dash-extensions>=1.0.4',
        'dash-bootstrap-components>=1.5.0',
        #'trace-updater>=0.0.9.1',
        'dash-vtk>=0.0.9',
        'dash>=2.14.1',
        'gmsh>=4.11.1',
        'meshio>=5.3.4',
        'scipy>=1.11.3',
        'pyvista>=0.42.3',
        'psycopg2>=2.9.9',
        'pyyaml>=6.0.2',
        'adios4dolfinx>=0.9.1.post0'
        ],
    packages = find_packages(),
    keywords = 'finite element method, thermal storage, simulation, optimization',
    python_requires='>3.10.0',
    classifiers=[
        'Natural Language :: English',
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3.10',
        'Operating System :: POSIX :: Linux',
    ]
)