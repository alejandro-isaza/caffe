#!/bin/bash

echo "$(tput setaf 2)"
echo "###################################################################"
echo "# Preparing to build LMDB"
echo "###################################################################"
echo "$(tput sgr0)"

# The results will be stored relative to the location
# where you stored this script, **not** relative to
# the location of the lmdb git repo.
PREFIX=`pwd`/..
if [ -d "${PREFIX}/platform" ]; then
    rm -rf "${PREFIX}/platform"
fi
mkdir -p "${PREFIX}/platform"

LMDB_VERSION=0.9.14
LMDB_RELEASE_URL=https://gitorious.org/mdb/mdb/archive/2f587ae081d076e3707360c5db086520c219d3ea.tar.gz
LMDB_RELEASE_DIRNAME=mdb-mdb

LMDB_SRC_DIR=/tmp/lmdb

# Uncomment if you want to see more information about each invocation
# of clang as the builds proceed.
# CLANG_VERBOSE="--verbose"

CC=clang
CXX=clang

CFLAGS="${CLANG_VERBOSE} -DNDEBUG -g -O0 -pipe -fPIC -fcxx-exceptions"
CXXFLAGS="${CLANG_VERBOSE} ${CFLAGS} -std=c++11 -stdlib=libc++"

LDFLAGS="-stdlib=libc++"
LIBS="-lc++ -lc++abi"

echo "PREFIX...................... ${PREFIX}"
echo "LMDB_VERSION ............... ${LMDB_VERSION}"
echo "LMDB_RELEASE_URL ........... ${LMDB_RELEASE_URL}"
echo "LMDB_RELEASE_DIRNAME ....... ${LMDB_RELEASE_DIRNAME}"
echo "LMDB_SRC_DIR ............... ${LMDB_SRC_DIR}"
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
    if [ -d ${LMDB_SRC_DIR} ]
    then
        rm -rf ${LMDB_SRC_DIR}
    fi

    cd `dirname ${LMDB_SRC_DIR}`

    if [ -d ${LMDB_RELEASE_DIRNAME} ]
    then
        rm -rf "${LMDB_RELEASE_DIRNAME}"
    fi
    curl --location ${LMDB_RELEASE_URL} --output ${LMDB_RELEASE_DIRNAME}.tar.gz
    tar xvf ${LMDB_RELEASE_DIRNAME}.tar.gz
    mv "${LMDB_RELEASE_DIRNAME}" "${LMDB_SRC_DIR}"
    rm ${LMDB_RELEASE_DIRNAME}.tar.gz
)

echo "$(tput setaf 2)"
echo "###################################################################"
echo "# Build"
echo "###################################################################"
echo "$(tput sgr0)"

(
    cd ${LMDB_SRC_DIR}/libraries/liblmdb

    mkdir -p ${PREFIX}/src/liblmdb
    cp mdb.c midl.c ${PREFIX}/src/liblmdb

    mkdir -p ${PREFIX}/include
    cp mdb.h midl.h ${PREFIX}/include
)

echo Done!

