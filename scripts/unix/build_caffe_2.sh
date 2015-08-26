#!/bin/sh
WORKDIR=/mnt/data

cd $WORKDIR/NN/caffe

# Setup with CMake
mkdir build
cd build
cmake .. -DBLAS=Open

# Fix glog issue
cd external/glog-prefix/src/glog
autoreconf -ivf
cd -

# Make
make -j8

# Get data
cd $WORKDIR/NN/data
wget -O Networks.zip https://www.dropbox.com/s/8gc8ouz06t8g0tm/Networks.zip\?dl\=1
wget -O Training.zip https://www.dropbox.com/s/zy62hyq3w926d71/Training.zip\?dl\=1
unzip Training.zip Networks.zip
rm -Rf __MACOSX
rm -rf Networks.zip Training.zip
mkdir kernels
mkdir snapshots
