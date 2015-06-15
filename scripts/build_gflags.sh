#!/bin/bash

PREFIX=`pwd`/..
GFLAGS_GIT_URL=https://github.com/gflags/gflags.git
GFLAGS_SRC_DIR=/tmp/gflags

if [ -d ${GFLAGS_SRC_DIR} ]
then
    rm -rf ${GFLAGS_SRC_DIR}
fi

cd `dirname ${GFLAGS_SRC_DIR}`
git clone ${GFLAGS_GIT_URL}
cd gflags
cmake .

if [ -d "${PREFIX}/src/gflags/" ]
then
    rm -rf "${PREFIX}/src/gflags/"
fi
mkdir -p "${PREFIX}/src/gflags"

if [ -d "${PREFIX}/include/gflags/" ]
then
    rm -rf "${PREFIX}/include/gflags/"
fi
mkdir -p "${PREFIX}/include/gflags"

cp include/gflags/gflags.h ${PREFIX}/include/gflags
cp include/gflags/gflags_declare.h ${PREFIX}/include/gflags
cp include/gflags/gflags_gflags.h ${PREFIX}/include/gflags
cp include/gflags/*.h ${PREFIX}/src/gflags
cp src/*.cc ${PREFIX}/src/gflags
cp src/*.h ${PREFIX}/src/gflags
