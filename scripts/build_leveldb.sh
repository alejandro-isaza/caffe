#!/bin/bash

echo "$(tput setaf 2)"
echo "###################################################################"
echo "# Preparing to build Google LevelDB"
echo "###################################################################"
echo "$(tput sgr0)"

# The results will be stored relative to the location
# where you stored this script, **not** relative to
# the location of the leveldb git repo.
PREFIX_IOS=`pwd`/../ios
PREFIX_OSX=`pwd`/../osx
if [ -d "${PREFIX_IOS}/platform" ]; then
    rm -rf "${PREFIX_IOS}/platform"
fi
mkdir -p "${PREFIX_IOS}/platform"

if [ -d "${PREFIX_OSX}/platform" ]; then
    rm -rf "${PREFIX_OSX}/platform"
fi
mkdir -p "${PREFIX_OSX}/platform"

LEVELDB_VERSION=1.18
LEVELDB_RELEASE_URL=https://github.com/google/leveldb/archive/v${LEVELDB_VERSION}.tar.gz
LEVELDB_RELEASE_DIRNAME=leveldb-${LEVELDB_VERSION}

LEVELDB_SRC_DIR=/tmp/leveldb

# Uncomment if you want to see more information about each invocation
# of clang as the builds proceed.
# CLANG_VERBOSE="--verbose"

CC=clang
CXX=clang

CFLAGS="${CLANG_VERBOSE} -DNDEBUG -g -O0 -pipe -fPIC -fcxx-exceptions"
CXXFLAGS="${CLANG_VERBOSE} ${CFLAGS} -std=c++11 -stdlib=libc++"

LDFLAGS="-stdlib=libc++"
LIBS="-lc++ -lc++abi"

echo "PREFIX iOS ..................... ${PREFIX_IOS}"
echo "PREFIX OSX ..................... ${PREFIX_OSX}"
echo "LEVELDB_VERSION ........... ${LEVELDB_VERSION}"
echo "LEVELDB_RELEASE_URL ....... ${LEVELDB_RELEASE_URL}"
echo "LEVELDB_RELEASE_DIRNAME ... ${LEVELDB_RELEASE_DIRNAME}"
echo "LEVELDB_SRC_DIR ........... ${LEVELDB_SRC_DIR}"
echo "CC ......................... ${CC}"
echo "CFLAGS ..................... ${CFLAGS}"
echo "CXX ........................ ${CXX}"
echo "CXXFLAGS ................... ${CXXFLAGS}"
echo "LDFLAGS .................... ${LDFLAGS}"
echo "LIBS ....................... ${LIBS}"

echo "$(tput setaf 2)"
echo "###################################################################"
echo "# Fetch Google LevelDB"
echo "###################################################################"
echo "$(tput sgr0)"

(
    if [ -d ${LEVELDB_SRC_DIR} ]
    then
        rm -rf ${LEVELDB_SRC_DIR}
    fi

    cd `dirname ${LEVELDB_SRC_DIR}`

    if [ -d ${LEVELDB_RELEASE_DIRNAME} ]
    then
        rm -rf "${LEVELDB_RELEASE_DIRNAME}"
    fi
    curl --location ${LEVELDB_RELEASE_URL} --output ${LEVELDB_RELEASE_DIRNAME}.tar.gz
    tar xvf ${LEVELDB_RELEASE_DIRNAME}.tar.gz
    mv "${LEVELDB_RELEASE_DIRNAME}" "${LEVELDB_SRC_DIR}"
    rm ${LEVELDB_RELEASE_DIRNAME}.tar.gz
)

echo "$(tput setaf 2)"
echo "###################################################################"
echo "# Build"
echo "###################################################################"
echo "$(tput sgr0)"

(
    cd ${LEVELDB_SRC_DIR}
    make PLATFORM=IOS
    mkdir -p ${PREFIX_IOS}/lib
    cp -f libleveldb.a ${PREFIX_IOS}/lib
    cp -R include ${PREFIX_IOS}

    make
    mkdir -p ${PREFIX_OSX}/lib
    cp -f libleveldb.a ${PREFIX_OSX}/lib
    cp -R include ${PREFIX_OSX}
)

echo Done!

