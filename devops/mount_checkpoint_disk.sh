# lsblk
# sudo mkdir /home/jupyter
# sudo mkfs.ext4 /dev/nvme0n2
# sudo mount /dev/nvme0n2 /home/jupyter
# rm ./checkpoints

# ./checkpoints directory
sudo mkdir /home/jupyter/checkpoints
ln -s /home/jupyter/checkpoints ./checkpoints
sudo chown -R $(whoami):$(whoami) /home/jupyter/checkpoints
sudo chmod 766 /home/jupyter/checkpoints

# ./artifacts directory
sudo mkdir /home/jupyter/artifacts
ln -s /home/jupyter/artifacts ./artifacts
sudo chown -R $(whoami):$(whoami) /home/jupyter/artifacts
sudo chmod 766 /home/jupyter/artifacts
