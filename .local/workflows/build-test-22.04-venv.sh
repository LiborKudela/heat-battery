### test build on ubuntu 22.04 ###
version="22.04"
name="jammy-jellyfish"
vm_name="vm-$name"
vm_user="ubuntu"
workspace_dir="/home/$vm_user/workspace"
bundle_name="heat-battery.bundle"
git_root_dir=$(git rev-parse --show-toplevel)
bundle_file_name=temp_bundle-$name.tar
venv_dir="$workspace_dir/hb_venv"
config_ppass=$(cat .github/github_test_config.yaml | grep -oP '(?<=password: )[^ ]+')
config_ppassc=$(echo ${config_ppass})
config_hbdir=$(cat .github/github_test_config.yaml | grep -oP '(?<=bin_files_folder_prefix: )[^ ]+')

exit_gracefully() {
    echo "Exiting gracefully..."
    rm -f $bundle_file_name
    exit 1
}

if [ "$1" = "--no-cache" ]; then
    echo "Removing previous VM $vm_name... (--no-cache)"
    multipass stop $vm_name
    multipass delete --purge $vm_name
fi

if ! multipass list | grep -q $vm_name; then
    echo "Creating new VM $vm_name..."
    multipass launch $version --name $vm_name --cpus 4 --memory 16384M --disk 20G
    multipass exec $vm_name -- bash -c "echo avail_df[\$(date '+%d.%m.%Y %H:%M:%S')]=\$(df . --output=avail | tail -n 1) > vminfo.initial"
fi

# create workspace directory and copy git files to it
multipass exec $vm_name -- mkdir -p $workspace_dir || exit_gracefully
multipass exec $vm_name -- cd $workspace_dir || exit_gracefully
files_to_copy=$(git ls-files --exclude-standard && git ls-files --others --exclude-standard)
files_to_copy_du=$(du -ch $files_to_copy | sort -rh)
top_files_sizewise=$(head -n 5 <<< "$files_to_copy_du")
echo "Top 5 files sizewise:"
echo "$top_files_sizewise"
echo "Copying files to bundle..."
tar -cf $bundle_file_name -T <(echo "$files_to_copy") || exit_gracefully
echo "Removing existing files from $name workspace..."
multipass exec $vm_name -- rm -rf $workspace_dir/* || exit_gracefully
echo "Transferring bundle to VM $name..."
multipass transfer $bundle_file_name $vm_name:$workspace_dir || exit_gracefully
rm $bundle_file_name
echo "Extracting bundle in VM $name..."
multipass exec $vm_name -- tar -xf $workspace_dir/$bundle_file_name -C $workspace_dir || exit_gracefully
echo "Changing directory to $name workspace..."
multipass exec $vm_name -- cd "$workspace_dir" || exit_gracefully
echo "Listing files in $name workspace..."
multipass exec $vm_name -- bash -c "cd $workspace_dir && ls -la" || exit_gracefully

# job steps follow here

#install python3-venv
multipass exec $vm_name -- sudo apt install -y python3-venv || exit_gracefully

#setup virtual environment if not already setup
if [ $venv_dir != "" ]; then
    echo "Setting up virtual environment..."
    if [ -d $venv_dir ]; then
        echo "Virtual environment already exists, skipping..."
    else
        echo "Creating venv: $venv_dir"
        multipass exec $vm_name -- python3 -m venv --system-site-packages $venv_dir || exit_gracefully
    fi
    activate_venv_cmd="source $venv_dir/bin/activate"
    deactivate_venv_cmd="deactivate"
    multipass exec $vm_name -- $activate_venv_cmd || exit_gracefully
else
    echo "Virtual environment not specified, skipping..."
    activate_venv_cmd=":"
    deactivate_venv_cmd=":"
fi

# run install script
echo "Running install script..."
multipass exec $vm_name -- bash -c "\
    cd $workspace_dir && \
    $activate_venv_cmd && \
    bash install_scripts/install_ubuntu.sh -y -p --ppass $config_ppass --ppassc $config_ppassc --hbdir $config_hbdir && \
    $deactivate_venv_cmd \
    " || exit_gracefully

# install coverage
echo "Installing coverage..."
multipass exec $vm_name -- bash -c "\
    cd $workspace_dir && \
    $activate_venv_cmd && \
    pip3 install coverage && \
    $deactivate_venv_cmd \
    " || exit_gracefully

# run tests
multipass exec $vm_name -- bash -c "\
    cd $workspace_dir && \
    $activate_venv_cmd && \
    python3 -m coverage run --source='./heat_battery' -m unittest discover -s test/ -p 'test_*.py' &&\
    $deactivate_venv_cmd \
    " || exit_gracefully

# report coverage
multipass exec $vm_name -- bash -c "\
    cd $workspace_dir && \
    $activate_venv_cmd && \
    python3 -m coverage report -m && \
    $deactivate_venv_cmd \
    " || exit_gracefully

# report coverage percentage
multipass exec $vm_name -- bash -c "\
    cd $workspace_dir && \
    $activate_venv_cmd && \
    value=\$(python3 -m coverage report -m | grep 'TOTAL' | awk '{print \$NF}' | sed 's/%//') && \
    echo \"Code coverage: \$value%\" && \
    $deactivate_venv_cmd \
    " || exit_gracefully

# report of disk usage
multipass exec $vm_name -- bash -c "echo avail_df[\$(date '+%d.%m.%Y %H:%M:%S')]=\$(df . --output=avail | tail -n 1) > vminfo.final"
initial_avail_space=$(multipass exec $vm_name -- bash -c "cat vminfo.initial | grep '^avail_df' | sed 's/.*=\s*\(.*\)/\1/'")
echo "Initial avail space: $initial_avail_space"
final_avail_space=$(multipass exec $vm_name -- bash -c "cat vminfo.final | grep '^avail_df' | sed 's/.*=\s*\(.*\)/\1/'")
echo "Final avail space: $final_avail_space"
consumed_space_raw=$(($initial_avail_space - $final_avail_space))
echo "Consumed space raw: $consumed_space_raw"
consumed_space_human=$(numfmt --to=si $(($consumed_space_raw*1000))) || echo "Failed to convert to human readable space"
echo "Consumed space human: $consumed_space_human"
exit 0


