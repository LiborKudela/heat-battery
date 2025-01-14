#! /bin/bash

# Function to display help message
Help() {
    echo
    echo "Install heat-battery package, its dependencies and setup PostgreSQL"
    echo
    echo "Options:"
    echo "  -y        Automatically answer yes to all apt prompts"
    echo "  -p        Also install and configure PostgreSQL"
    echo "  --help    Show this help message"
    echo "  --ppass   Set PostgreSQL password for postgres user"
    echo "  --ppassc  Confirm PostgreSQL password for postgres user (must be the same as ppass)"
    echo "  --hbdir   Set heat battery data directory (default: /home/username/heat_battery_data)"
    echo
    echo "IMPORTANT:"
    echo "  The options --ppass and --ppassc are required when -p is set."
    echo
    echo "Example:"
    echo "  $0 -y"
    echo "  $0 -y -p --ppass mypassword --ppassc mypassword --hbdir /home/user/heat_battery_data"
    echo
    echo "The script will:"
    echo "  1. Update system packages"
    echo "  2. Install and test OpenMPI"
    echo "  3. Install and test Gmsh"
    echo "  4. Install FEniCSx"
    echo "  5. Build, install and test ADIOS2 with MPI enabled"
    echo "  6. Install and configure PostgreSQL if -p is set"
    echo "  7. Install heat-battery python3 module"
    exit 0
}

# default arguments values
user_name=$(whoami)
auto_yes=""
auto_yes_bool=false
install_postgres=false
postgres_password=""
postgres_password_confirm=""
heat_battery_data_dir="home/$user_name/heat_battery_data"

# parse options
VALID_ARGS=$(getopt -o yp --long ppass:,ppassc:,hbdir: -- "$@")

# exit if getopt fails
if [ $? -ne 0 ]; then
    Help
    exit 1;
fi

# count non option arguments
no_option_args=0
skip_next=false; 
for arg in $VALID_ARGS; do
    if [[ "$arg" = -- ]]; then
        skip_next=false;
    elif [[ "$arg" = --* ]]; then
        skip_next=true;
    elif [[ "$arg" = -* ]]; then
        skip_next=false;
        :
    else
        if [[ $skip_next = false ]]; then
            no_option_args=$((no_option_args + 1))
            skip_next=false;
        else
            :
        fi
    fi
done

# # exit if there are positional arguments 
# coment this block if you want to use positional arguments as input ect.
if [ $no_option_args -gt 0 ]; then
    echo "Error: Arguments that are not defined in the help are not allowed."
    Help
    exit 1
fi

# make sure we run the fresh set from getopt
eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    -y)
        auto_yes="-y"
        auto_yes_bool=true
        shift
        ;;
    -p)
        install_postgres=true
        shift
        ;;
    --help)
        Help
        ;;
    --ppass)
        postgres_password=$2
        shift 2
        ;;
    --ppassc)
        postgres_password_confirm=$2
        shift 2
        ;;
    --hbdir)
        heat_battery_data_dir=$2
        shift 2
        ;;
    --) shift; 
        break 
        ;;
  esac
done

# check if arguments are valid
if $install_postgres; then
    echo "ps set"
    if [ "$postgres_password" != "" ]; then
        if [ "$postgres_password" != "$postgres_password_confirm" ]; then
            echo "PostgreSQL password and confirmation password do not match! Exiting.."
            Help;
            exit 1
        fi
    else
        echo "PostgreSQL password is required when -p is set! Exiting.."
        Help;
        exit 1
    fi
fi

echo "Running script with options:"
echo "  -y: $auto_yes_bool"
echo "  -p: $install_postgres"
echo "  --ppass: $postgres_password"
echo "  --ppassc: $postgres_password_confirm"
echo "  --hbdir: $heat_battery_data_dir"

echo "Inputs: $no_option_args"
if [ $no_option_args -gt 0 ]; then
    echo "Inputs:"
    for arg in "$@"; do
        echo "  $arg"
    done
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
sudo apt upgrade $auto_yes
echo "APT available packages list updated!"
sudo apt install build-essential $auto_yes
sudo apt install cmake $auto_yes
sudo apt install git $auto_yes
sudo apt install python3-dev $auto_yes
sudo apt install python3-pip $auto_yes
sudo apt install tree $auto_yes
echo "Build-essential and python3-pip installed!"

# get python3 version and current working directory
PY_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
ORG_PWD=$(pwd)

# ask for password for postgres user
if [ "$install_postgres" = "true" ]; then
    echo "POSTGRES server packages will be installed now!"
    sudo apt install postgresql postgresql-contrib $auto_yes
    sudo apt install postgresql-plpython3 $auto_yes || plpython3_failed=true
    echo "try_postgres_ppa: $TRY_POSTGRESQL_PPA" || echo "try_postgres_ppa: false"
    if [ "$plpython3_failed" = "true" ] && [ "$TRY_POSTGRESQL_PPA" = "true" ]; then
        echo "Failed to install postgresql-plpython3, trying to install plpython3 from PostgreSQL PPA..."
        echo "Installing postgresql-common..."
        sudo apt install $auto_yes postgresql-common
        echo "Installing PostgreSQL PPA..."
        sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
        sudo apt update
        echo "Available postgresql-plpython3 packages at this moment:"
        sudo apt-cache madison postgresql-plpython3
        echo "Attempting to install postgresql-plpython3 again..."
        sudo apt install $auto_yes postgresql-plpython3|| apt install $auto_yes postgresql-plpython3=14.15-1.pgdg22.04+1 || exit 1
    fi
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
    sudo pg_isready
    echo "PostgreSQL server started successfully!"

    # set password for postgres user
    echo "Setting password for postgres user..."
    sudo -u postgres psql template1 -c "ALTER USER postgres with encrypted password '$postgres_password';"
    echo "Password for postgres user set successfully!"

    # set permissions for postgres user
    echo "Setting permissions for postgres user..."
    sudo mkdir -p $heat_battery_data_dir
    sudo setfacl -Rm u:postgres:rwx,u:$(whoami):rwx $heat_battery_data_dir
    sudo setfacl -Rdm u:postgres:rwx,u:$(whoami):rwx $heat_battery_data_dir
    echo "Permissions for postgres user set successfully!"
fi

# install openmpi
# echo "Instaling OpenMPI and mpi4py!"
# sudo apt install openmpi-bin openmpi-doc libopenmpi-dev $auto_yes
# pip3 install mpi4py
# if ! mpirun -n 1 python3 -c "from mpi4py import MPI; print(f'rank: {MPI.COMM_WORLD.rank}')"; then
#     echo "Failed to run mpi4py"
#     exit 1
# fi
# echo "OpenMPI installed!"

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
    paths=(
        "lib/python${PY_VERSION}/dist-packages/adios2" 
        "local/lib/python${PY_VERSION}/dist-packages/adios2"
        "lib/python3/dist-packages/adios2"
        "local/lib/python3/dist-packages/adios2" 
    )
    for src_path in "${paths[@]}"; do
        echo "Checking python package at path: $src_path"
        if [ -d "$src_path" ]; then
            echo "Python package data FOUND at $src_path"
            echo "Copying adios2 bindings from $src_path to ${p3_corect_dir}"
            sudo mkdir -p ${p3_corect_dir}
            sudo cp -r "$src_path"/* ${p3_corect_dir}/
            break
        else
            echo "Python package data NOT FOUND at $src_path"
        fi
    done
    sudo ldconfig
    echo "Trying again python import again..."
    if ! python3 -c "import adios2; print(f'adios2 version: {adios2.__version__}')"; then
        tree
        echo "Python adios2 import test has failed, see tree of build-adios2 directory above - exiting..."
        exit 1
    fi
fi
echo "Python3 adios2 import test successful!"
cd $ORG_PWD
rm -rf ADIOS2
rm -rf build-adios2
echo "ADIOS2 installed!"

# install libpg for worker nodes
echo "Installing libpq-dev (needed by worker nodes)!"
sudo apt install libpq-dev $auto_yes
echo "libpq-dev installed!"

# install heat_battery
echo "Installing heat_battery python package..."
pip3 install .
if ! python3 -c "import heat_battery; print('heat_battery imported successfully!')"; then
    echo "Failed to import heat_battery"
    exit 1
fi
echo "HeatBattery installed successfully!"

