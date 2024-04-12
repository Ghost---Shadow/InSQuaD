lsblk
sudo mkdir /mnt/checkpoints
sudo mkfs.ext4 /dev/nvme0n2 # Needs to be done only once
sudo mount /dev/nvme0n2 /mnt/checkpoints
# rm ./checkpoints

# ./checkpoints directory
sudo mkdir /mnt/checkpoints/checkpoints
ln -s /mnt/checkpoints/checkpoints ./checkpoints
sudo chown -R $(whoami):$(whoami) /mnt/checkpoints/checkpoints
sudo chmod 766 /mnt/checkpoints/checkpoints

# ./artifacts directory
sudo mkdir /mnt/checkpoints/artifacts
ln -s /mnt/checkpoints/artifacts ./artifacts
sudo chown -R $(whoami):$(whoami) /mnt/checkpoints/artifacts
sudo chmod 766 /mnt/checkpoints/artifacts

# Send to fstab in case of reboot
LINE='/dev/nvme0n2 /mnt/checkpoints ext4 defaults 0 2'

if ! grep -Fxq "$LINE" /etc/fstab; then
    echo "$LINE" | sudo tee -a /etc/fstab
else
    echo "$LINE already exists in /etc/fstab"
fi

lsblk
