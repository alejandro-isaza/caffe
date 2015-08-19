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

AQUILA_RELEASE_DIRNAME=aquila-src
AQUILA_RELEASE_URL=git://github.com/zsiciarz/aquila.git\ ${AQUILA_RELEASE_DIRNAME}
AQUILA_LIB=libAquila.a
OOURA_LIB=lib/libOoura_fft.a

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
echo "AQUILA_VERSION ............... ${AQUILA_VERSION}"
echo "AQUILA_RELEASE_URL ........... ${AQUILA_RELEASE_URL}"
echo "AQUILA_RELEASE_DIRNAME ....... ${AQUILAAQUILA_RELEASE_DIRNAME}"
echo "AQUILA_SRC_DIR ............... ${AQUILA_SRC_DIR}"
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
    if [ -d ${AQUILA_RELEASE_DIRNAME} ]
    then
        rm -rf "${AQUILA_RELEASE_DIRNAME}"
    fi
    git clone ${AQUILA_RELEASE_URL}
)

echo "$(tput setaf 2)"
echo "###################################################################"
echo "# Build"
echo "###################################################################"
echo "$(tput sgr0)"
(
    cd ${AQUILA_RELEASE_DIRNAME}
    cmake .
    make

    mv "${AQUILA_LIB}" "../lib/"
    mv "${OOURA_LIB}" "../lib/"
    mv "aquila" "../include/"

    cd ..
    rm -fr ${AQUILA_SRC_DIR}
)

echo Done!
