# sudo apt remove linux-image-5.10.0-29-cloud-amd64
# sudo apt-mark hold linux-image-5.10.0-27-cloud-amd64
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
sudo python3 install_gpu_driver.py verify 
