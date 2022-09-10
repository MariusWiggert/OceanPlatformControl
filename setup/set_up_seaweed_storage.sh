# Step 1: Install sshfs
if which sshfs > /dev/null; then
    echo -e "\033[0;32msshfs installed.\033[0m"
else
    echo -e "\033[0;33msshfs not installed.\033[0m"
    sudo apt-get install -y sshfs
fi

# Step 2: Create Folder
if [ -d "/seaweed-storage" ]; then
    echo -e "\033[0;32mDirectory /seaweed-storage already exists.\033[0m"
else
    echo -e "\033[0;33mCreate directory /seaweed-storage.\033[0m"
    sudo mkdir -p /seaweed-storage
    sudo chown ubuntu /seaweed-storage
    sudo chmod 777 /seaweed-storage
fi

# Step 3: Mount Folder
if [ -f "/seaweed-storage/connected" ]; then
    echo -e "\033[0;32mSeaweed Storage already connected.\033[0m"
else
    echo -e "\033[0;33mReconnecting Seaweed Storage.\033[0m"
    sudo fusermount -q -u /seaweed-storage
    sshfs -o ssh_command='ssh -i /home/ubuntu/OceanPlatformControl/setup/azure -o StrictHostKeyChecking=no' ubuntu@20.55.80.215:/seaweed-storage /seaweed-storage -o default_permissions

    if [ -f "/seaweed-storage/connected" ]; then
        echo -e "\033[0;32m/seaweed-storage reconnected.\033[0m"
    else
        echo -e "\033[0;31m/seaweed-storage could not be connected.\033[0m"
    fi
fi