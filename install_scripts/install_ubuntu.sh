#! /bin/bash

# Function to display help message
Help() {
    echo "Usage: $0 [-p install_postgres] [-ppass postgres_password]"
    echo
    echo "Install heat-battery package, its dependencies and setup PostgreSQL"
    echo
    echo "Options:"
    echo "  -y       Automatically answer yes to all apt-get prompts"
    echo "  -p       Also install and configure PostgreSQL"
    echo "  -ppass   Set PostgreSQL password for postgres user"
    echo "  -ppassc  Confirm PostgreSQL password for postgres user"
    echo "  -hbdir   Set heat battery data directory (default: /home/username/heat_battery_data)"
    echo
    echo "IMPORTANT:"
    echo "  The options -ppass and -ppassc are required when -p is set."
    echo
    echo "Example:"
    echo "  $0 -p -ppass mypassword -ppassc mypassword -hbdir /home/user/heat_battery_data"
    echo
    echo "The script will:"
    echo "  1. Update system packages"
    echo "  2. Install and test OpenMPI"
    echo "  3. Install and test Gmsh"
    echo "  4. Install FEniCSx"
    echo "  5. Install and configure PostgreSQL if -p is set"
    echo "  6. Install heat-battery package"
}
# default arguments
user_name=$(whoami)
auto_yes=""
install_postgres=0
postgres_password=""
postgres_password_confirm=""
heat_battery_data_dir="home/$user_name/heat_battery_data"

#parse bash named arguments
while getopts ":p:ppass:ppassc:hbdir:" opt; do
    case $opt in
        y ) auto_yes="-y";;
        p ) install_postgres=1;;
        ppass ) postgres_password=$OPTARG;;
        ppassc ) postgres_password_confirm=$OPTARG;;
        hbdir ) heat_battery_data_dir=$OPTARG;;
        \? ) Help; exit 1;;
    esac
done

# check if arguments are valid
if [ "$install_postgres" = "1" ] then
    if [ "$postgres_password" != "" ]; then
        if [ "$postgres_password" != "$postgres_password_confirm" ]; then
            echo "PostgreSQL password and confirmation password do not match! Exiting.."
            exit 1
        fi
    else
        echo "PostgreSQL password is required when -p is set! Exiting.."
        exit 1
    fi
fi

#check if heat_batter_data_dir parent directory exists when default is used
if [ "$heat_battery_data_dir" = "home/$user_name/heat_battery_data" ]; then
    if [ ! -d "/home/$user_name" ]; then
        echo "Parent directory /home/$user_name does not exist! Exiting.."
        echo "Cannot set default heat_battery_data_dir to /home/$user_name/heat_battery_data"
        exit 1
    fi
fi

echo "Provided argumets are valid, starting installation..."

# update system and ensure essential packages are installed
sudo apt update
sudo apt upgrade
echo "APT available packages list updated!"
sudo apt install build-essential $auto_yes
sudo apt install cmake $auto_yes
sudo apt install git $auto_yes
sudo apt install python3-pip $auto_yes
echo "Build-essential and python3-pip installed!"

# get python3 version and current working directory
PY_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
ORG_PWD=$(pwd)

# install openmpi
echo "Instaling OpenMPI and mpi4py!"
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev $auto_yes
pip3 install mpi4py
if ! mpirun -n 1 python3 -c "from mpi4py import MPI; print(f'rank: {MPI.COMM_WORLD.rank}')"; then
    echo "Failed to run mpi4py"
    exit 1
fi
echo "OpenMPI installed!"

# install gmsh
echo "Installing Gmsh and gmsh python package!"
sudo apt install gmsh $auto_yes
pip3 install gmsh
if ! python3 -c "import gmsh"; then
    echo "Failed to import gmsh"
    exit 1
fi
echo "Gmsh installed!"

# install fenicsx
echo "Installing FEniCSx!"
sudo add-apt-repository ppa:fenics-packages/fenics $auto_yes
sudo apt update
sudo apt install fenicsx $auto_yes
if ! python3 -c "import dolfinx; print(f'dolfinx version: {dolfinx.__version__}')"; then
    echo "Failed to import dolfinx"
    exit 1
fi
echo "Fenicsx installed!"

# build and install adios2
echo "Building and installing adios2!"
git clone https://github.com/ornladios/ADIOS2.git ADIOS2
mkdir build-adios2
cd build-adios2
cmake ../ADIOS2 -DADIOS2_BUILD_EXAMPLES=ON
make -j $(nproc)
sudo make install
sudo ldconfig

# sometimes the make install copies the adios2 binding to wrong directory
if ! python3 -c "import adios2; print(f'adios2 version: {adios2.__version__}')"; then

    p3_corect_dir="/usr/local/lib/python${PY_VERSION}/dist-packages/adios2"
    echo "Python adios2 import failed - copying binding directly to ${p3_corect_dir}"
    sudo mkdir -p ${p3_corect_dir}
    sudo cp -r /lib/python3/dist-packages/adios2* ${p3_corect_dir}/
    sudo ldconfig
    echo "Trying again python import again..."
    if ! python3 -c "import adios2; print(f'adios2 version: {adios2.__version__}')"; then
        echo "Python adios2 import test has failed - exiting..."
        exit 1
    fi
fi
echo "Python3 adios2 import test successful!"
cd $ORG_PWD
rm -rf ADIOS2
rm -rf build-adios2
echo "ADIOS2 installed!"

# install libpg for worker nodes
sudo apt install libpq-dev $auto_yes
echo "libpq-dev installed (needed by worker nodes)!"

# ask for password for postgres user
if [ "$install_postgres" = "true" ]; then
    echo "POSTGRES server packages will be installed now!"
    sudo apt install postgresql postgresql-contrib $auto_yes
    sudo apt install postgresql-plpython3-14 $auto_yes
    sudo apt install acl $auto_yes
    echo "PostgreSQL server packages installed!"

    echo "Starting PostgreSQL server..."
    # on normal Ubuntu
    if command -v systemctl >/dev/null 2>&1; then
        sudo systemctl start postgresql.service

    # on WSL Ubuntu
    elif command -v service >/dev/null 2>&1; then
        sudo service postgresql start

    else
        echo "Could not start PostgreSQL service!"
        echo "No command in this list found:"
        echo "   - systemctl (for normal Ubuntu)"
        echo "   - service (for WSL Ubuntu)"
        exit 1
    fi
    echo "PostgreSQL server started successfully!"

    # set password for postgres user
    echo "Setting password for postgres user..."
    sudo -u postgres psql template1 -c "ALTER USER postgres with encrypted password '$postgres_password';"
    echo "Password for postgres user set successfully!"

    # set permissions for postgres user
    echo "Setting permissions for postgres user..."
    sudo setfacl -Rm u:postgres:rwx,u:$(whoami):rwx $heat_battery_data_dir
    sudo setfacl -Rdm u:postgres:rwx,u:$(whoami):rwx $heat_battery_data_dir
    echo "Permissions for postgres user set successfully!"
fi

# install heat_battery
echo "Installing heat_battery python package..."
pip3 install .
if ! python3 -c "import heat_battery; print('heat_battery imported successfully!')"; then
    echo "Failed to import heat_battery"
    exit 1
fi
echo "HeatBattery installed successfully!"

