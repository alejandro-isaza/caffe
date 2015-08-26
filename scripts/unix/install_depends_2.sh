#!/bin/sh
WORKDIR=/mnt/data

cd $WORKDIR/nvidia_installers
sudo ./NVIDIA-Linux-x86_64-346.46.run
# See https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-(Ubuntu,-CUDA-7,-cuDNN)

# Install Aquila
cd $WORKDIR
git clone git://github.com/zsiciarz/aquila.git aquila-src
cd aquila-src
mkdir build && cd build && cmake ..
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
