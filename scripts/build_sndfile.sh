#!/bin/bash

echo "$(tput setaf 2)"
echo "###################################################################"
echo "# Preparing to build Aquila"
echo "###################################################################"
echo "$(tput sgr0)"

# The results will be stored relative to the location
# where you stored this script, **not** relative to
# the location of the AQUILA git repo.
PREFIX=`pwd`/..
if [ -d "${PREFIX}/platform" ]; then
    rm -rf "${PREFIX}/platform"
fi
mkdir -p "${PREFIX}/platform"

LIBSND_VERSION=1.0.25
LIBSND_SRC_DIRNAME=libsndfile-${LIBSND_VERSION}
LIBSND_RELEASE_URL=http://www.mega-nerd.com/libsndfile/files/libsndfile-${LIBSND_VERSION}.tar.gz

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
echo "AQUILA_VERSION ............... ${LIBSND_VERSION}"
echo "LIBSND_RELEASE_URL ........... ${LIBSND_RELEASE_URL}"
echo "LIBSND_SRC_DIRNAME ....... ${LIBSND_SRC_DIRNAME}"
echo "CC ......................... ${CC}"
echo "CFLAGS ..................... ${CFLAGS}"
echo "CXX ........................ ${CXX}"
echo "CXXFLAGS ................... ${CXXFLAGS}"
echo "LDFLAGS .................... ${LDFLAGS}"
echo "LIBS ....................... ${LIBS}"

echo "$(tput setaf 2)"
echo "###################################################################"
echo "# Fetch AQUILA"
echo "###################################################################"
echo "$(tput sgr0)"

(
    if [ -d ${LIBSND_SRC_DIRNAME} ]
    then
        rm -rf "${LIBSND_SRC_DIRNAME}"
    fi
    curl -L ${LIBSND_RELEASE_URL} -o ${LIBSND_SRC_DIRNAME}.tar.gz
    tar -zxvf ${LIBSND_SRC_DIRNAME}.tar.gz

    rm -rf ${LIBSND_SRC_DIRNAME}.tar.gz
)

echo "$(tput setaf 2)"
echo "###################################################################"
echo "# Build"
echo "###################################################################"
echo "$(tput sgr0)"
(
    cd ${LIBSND_SRC_DIRNAME}
    ./configure
    make
    sudo make install

    cd ..
    rm -fr ${LIBSND_SRC_DIRNAME}
)

echo Done!
