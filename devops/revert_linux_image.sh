#!/bin/bash

# Desired kernel version
desired_version="5.10.0-27-cloud-amd64"

# Install the desired version if it's not already installed
if ! dpkg --list | grep -q "linux-image-$desired_version"; then
    echo "Installing linux-image-$desired_version..."
    sudo apt install "linux-image-$desired_version" -y
fi

# Mark the installed version so it won't be upgraded
sudo apt-mark hold "linux-image-$desired_version"

# Remove all kernel versions greater than the base version of 5.10.0-27
base_version="5.10.0-27"
dpkg --list | grep 'linux-image' | awk '{print $2}' | while read image; do
    # Extract the full version number from the package name
    version=$(echo "$image" | grep -oP 'linux-image-\K.*')

    # Check if this full version is greater than the base version
    if dpkg --compare-versions "$version" gt "$base_version"; then
        echo "Removing $image..."
        sudo apt remove "$image" -y
    fi
done

# Update GRUB to reflect the kernel changes
sudo update-grub

# Disable automatic updates script
source devops/disable_updates.sh
