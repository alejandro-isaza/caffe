#!/bin/sh
WORKDIR=/mnt/data/NN/caffe

mkdir $WORKDIR/tools/disabled $WORKDIR/src/caffe/layers/disabled $WORKDIR/src/caffe/util/disabled

mv $WORKDIR/src/caffe/layers/hdf5_data_layer.cpp $WORKDIR/src/caffe/layers/disabled/hdf5_data_layer.cpp
mv $WORKDIR/src/caffe/layers/hdf5_data_layer.cu $WORKDIR/src/caffe/layers/disabled/hdf5_data_layer.cu
mv $WORKDIR/src/caffe/layers/hdf5_output_layer.cpp $WORKDIR/src/caffe/layers/disabled/hdf5_output_layer.cpp
mv $WORKDIR/src/caffe/layers/hdf5_output_layer.cu $WORKDIR/src/caffe/layers/disabled/hdf5_output_layer.cu
mv $WORKDIR/src/caffe/layers/image_data_layer.cpp $WORKDIR/src/caffe/layers/disabled/image_data_layer.cpp
mv $WORKDIR/src/caffe/layers/tri_slice_data_layer.cpp $WORKDIR/src/caffe/layers/disabled/tri_slice_data_layer.cpp
mv $WORKDIR/src/caffe/layers/window_data_layer.cpp $WORKDIR/src/caffe/layers/disabled/window_data_layer.cpp
mv $WORKDIR/src/caffe/util/read_audio_apple.cpp $WORKDIR/src/caffe/util/disabled/read_audio_apple.cpp
mv $WORKDIR/tools/computer_image_mean.cpp $WORKDIR/tools/disabled/computer_image_mean.cpp
