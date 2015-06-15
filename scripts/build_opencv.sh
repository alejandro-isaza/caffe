#!/bin/bash

LIBS_DIR=`pwd`/../lib
OPENCV_GIT_URL=https://github.com/Itseez/opencv.git
OPENCV_SRC_DIR=/tmp/opencv

if [ -d ${OPENCV_SRC_DIR} ]
then
    rm -rf ${OPENCV_SRC_DIR}
fi

cd `dirname ${OPENCV_SRC_DIR}`
git clone ${OPENCV_GIT_URL}

python opencv/platforms/ios/build_framework.py ios

if [ -d ${LIBS_DIR}/opencv2.framework ]
then
    rm -rf ${LIBS_DIR}/opencv2.framework
fi
cp -Rf ios/opencv2.framework ${LIBS_DIR}
