#! /bin/bash

# Function to display help message
Help() {
    echo "Usage: $0 [-p install_postgres] [-ppass postgres_password]"
    echo
    echo "Install heat-battery package and its dependencies"
    echo
    echo "Options:"
    echo "  -p    Also install and configure PostgreSQL"
    echo "  -ppass    Set PostgreSQL password for postgres user"
    echo
    echo "Example:"
    echo "  $0 -p true -ppass mypassword"
    echo
    echo "The script will:"
    echo "  1. Update system packages"
    echo "  2. Install OpenMPI and test mpi4py"
    echo "  3. Install Gmsh and test import" 
    echo "  4. Install FEniCSx"
    echo "  5. Install and configure PostgreSQL"
    echo "  6. Install heat-battery package"
}

# Show help if no arguments or -h/--help
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    Help
    exit 0
fi

# update system
sudo apt update
sudo apt install build-essential -y
sudo apt install python3-pip -y
echo "System updated!"

# install openmpi
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev -y
pip3 install mpi4py
mpirun -n 1 python3 -c "from mpi4py import MPI; print(f'rank: {MPI.COMM_WORLD.rank}')"
echo "OpenMPI installed!"

# install gmsh
sudo apt install gmsh -y
pip3 install gmsh
python3 -c "import gmsh"
echo "Gmsh installed!"

# install fenicsx
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt update
sudo apt install fenicsx -y
echo "Fenicsx installed!"
python3 -c "import dolfinx; print(f'dolfinx version: {dolfinx.__version__}')"

# install libpg for worker nodes
sudo apt install libpq-dev -y

# ask for password for postgres user
if [ "$install_postgres" = "true" ]; then
    read -p "Enter password for postgres user: " postgres_password
    read -p "Confirm password for postgres user: " postgres_password_confirm
    if [ "$postgres_password" != "$postgres_password_confirm" ]; then
        echo "Passwords do not match!"
        exit 1
    fi
    if [ "$postgres_password" != "" ]; then
        sudo apt install postgresql postgresql-contrib -y
        if command -v systemctl >/dev/null 2>&1; then
            sudo systemctl start postgresql.service
        elif command -v service >/dev/null 2>&1; then
            sudo service postgresql start
        else
            echo "Could not find systemctl or service command to start postgresql!"
            exit 1
        fi
        sudo -u postgres psql template1 -c "ALTER USER postgres with encrypted password '$postgres_password';"
    fi
fi

# install heat_battery
pip3 install .
python3 -c "import heat_battery; print('heat_battery imported successfully!')"
echo "HeatBattery installed!"

