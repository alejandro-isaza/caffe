#!/bin/sh
WORKDIR=/mnt/data

# Add -fPIC flag for Aquila
cd $WORKDIR/NN/caffe
echo "IF( CMAKE_SYSTEM_PROCESSOR STREQUAL \"x86_64\" )" >> CMakeLists.txt
echo "  SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS \"-fPIC std=c++11\")" >> CMakeLists.txt
echo "ENDIF( CMAKE_SYSTEM_PROCESSOR STREQUAL \"x86_64\" )" >> CMakeLists.txt

cd $WORKDIR/NN/caffe/scripts/unix
./remove_files.sh
./install_depends_1.sh
