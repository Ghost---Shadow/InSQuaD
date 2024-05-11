#!/bin/bash

# sudo bash devops/disable_updates.sh

# Disables automatic updates on a Debian-based system

# Check if the script is run as root
if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root. Use 'sudo' to run it."
  exit 1
fi

# Disable unattended upgrades by modifying the config file
echo "Disabling unattended upgrades..."
if [ -f /etc/apt/apt.conf.d/20auto-upgrades ]; then
  sed -i 's/^\(APT::Periodic::Update-Package-Lists\).*$/\1 "0";/' /etc/apt/apt.conf.d/20auto-upgrades
  sed -i 's/^\(APT::Periodic::Unattended-Upgrade\).*$/\1 "0";/' /etc/apt/apt.conf.d/20auto-upgrades
else
  echo "APT::Periodic::Update-Package-Lists \"0\";" > /etc/apt/apt.conf.d/20auto-upgrades
  echo "APT::Periodic::Unattended-Upgrade \"0\";" >> /etc/apt/apt.conf.d/20auto-upgrades
fi

# Stop and disable the unattended-upgrades service
echo "Stopping and disabling the unattended-upgrades service..."
systemctl stop unattended-upgrades
systemctl disable unattended-upgrades

# Check and comment out any update-related cron jobs in system-wide cron directories
echo "Checking for update-related cron jobs..."
CRON_DIRS=(/etc/cron.daily /etc/cron.weekly /etc/cron.monthly)
for dir in "${CRON_DIRS[@]}"; do
  for job in "$dir"/*; do
    if grep -q 'apt\|unattended-upgrades\|update' "$job"; then
      echo "Commenting out update-related job in $job"
      sed -i 's/^/#/' "$job"
    fi
  done
done

echo "Automatic updates have been disabled."
