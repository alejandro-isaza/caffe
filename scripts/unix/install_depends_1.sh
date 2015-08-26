#!/bin/sh
WORKDIR=/mnt/data

# Install packages
apt-get update
apt-get install build-essential checkinstall git cmake unzip
apt-get install libleveldb-dev libsnappy-dev libopencv-dev libboost-dev libhdf5-dev liblmdb-dev libsndfile1-dev libatlas-dev libopenblas-dev

# Install NVIDIA drivers
cd $WORKDIR
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run
chmod +x cuda_7.0.28_linux.run
mkdir nvidia_installers
./cuda_7.0.28_linux.run -extract=`pwd`/nvidia_installers
sudo apt-get install linux-image-extra-virtual
echo "blacklist nouveau" >> /etc/modprobe.d/blacklist-nouveau.conf
echo "blacklist lbm-nouveau" >> /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf
echo "alias nouveau off" >> /etc/modprobe.d/blacklist-nouveau.conf
echo "alias lbm-nouveau off" >> /etc/modprobe.d/blacklist-nouveau.conf

sudo apt-get install linux-source
sudo apt-get install linux-headers-`uname -r`

echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u
echo "******* REBOOTING: LOGIN AFTER REBOOT AND RUN NN/caffe/scripts/unix/build_caffe_2.sh"
wait 5000
sudo reboot
