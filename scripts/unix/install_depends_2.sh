#!/bin/sh
WORKDIR=/mnt/data

cd $WORKDIR/nvidia_installers
sudo ./NVIDIA-Linux-x86_64-346.46.run
# See https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-(Ubuntu,-CUDA-7,-cuDNN)

# cuFFT
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/cufft_update/cufft_patch_linux.tar.gz
cp cufft_patch_linux.tar.gz ~
cd /usr/local/cuda-7.0
sudo tar zxvf ~/cufft_patch_linux.tar.gz --keep-directory-symlink

# cuDNN
wget http://adgo.ca/cudnn-7.0-linux-x64-v3.0-rc.tar
tar -zxf cudnn-7.0-linux-x64-v3.0-rc.tar
cd cuda
sudo cp lib*/* /usr/local/cuda/lib64/
sudo cp include/cudnn.h /usr/local/cuda/include/

# Install Aquila
cd $WORKDIR
git clone git://github.com/zsiciarz/aquila.git aquila-src
cd aquila-src
mkdir build && cd build && cmake .. -DCMAKE_CXX_FLAGS="-fPIC"
make -j8
sudo make install
sudo cp lib/libOoura_fft.a /usr/local/lib/

# Install Protobuf
cd $WORKDIR
wget https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.bz2
tar xvjf protobuf-2.6.1.tar.bz2
cd protobuf-2.6.1
./autogen.sh
./configure --prefix=/usr
make -j8
make check
sudo make install
