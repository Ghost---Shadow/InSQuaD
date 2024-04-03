#!/bin/bash

# URL of the Miniconda installer script
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

# File name for the downloaded Miniconda installer
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"

# Download Miniconda installer
echo "Downloading Miniconda installer..."
wget "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"

# Verify if download was successful
if [ $? -ne 0 ]; then
    echo "Failed to download Miniconda installer. Exiting..."
    exit 1
fi

# Run Miniconda installer
echo "Installing Miniconda..."
bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda"

# Verify if installation was successful
if [ $? -ne 0 ]; then
    echo "Failed to install Miniconda. Exiting..."
    exit 1
fi

# Add Miniconda to PATH
export PATH="$HOME/miniconda/bin:$PATH"

# Cleanup downloaded installer
echo "Cleaning up..."
rm "$MINICONDA_INSTALLER"

echo "Miniconda installation complete."
