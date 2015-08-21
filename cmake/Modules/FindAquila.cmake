# - Find sndfile
# Find the native sndfile includes and libraries
#
#  AQUILA_INCLUDE_DIR - where to find sndfile.h, etc.
#  AQUILA_LIBRARY   - List of libraries when using libsndfile.
#  SNDFILE_FOUND       - True if libsndfile found.

FIND_PATH(OOURA_INCLUDE_DIR NAMES ooura/fft4g.c PATHS $ENV{LEVELDB_ROOT}/include /opt/local/include /usr/local/include /usr/include)
FIND_PATH(AQUILA_INCLUDE_DIR NAMES aquila/aquila.h PATHS $ENV{LEVELDB_ROOT}/include /opt/local/include /usr/local/include /usr/include)

FIND_LIBRARY(OOURA_LIBRARY NAMES Ooura_fft PATHS /usr/lib /usr/local/lib $ENV{LEVELDB_ROOT}/lib)
FIND_LIBRARY(AQUILA_LIBRARY NAMES Aquila PATHS /usr/lib /usr/local/lib $ENV{LEVELDB_ROOT}/lib)

MESSAGE( STATUS "OOURA_INCLUDE_DIR = \"${OOURA_INCLUDE_DIR}\"" )
MESSAGE( STATUS "OOURA_LIBRARY = \"${OOURA_LIBRARY}\"" )
MESSAGE( STATUS "AQUILA_INCLUDE_DIR = \"${AQUILA_INCLUDE_DIR}\"" )
MESSAGE( STATUS "AQUILA_LIBRARY = \"${AQUILA_LIBRARY}\"" )


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AQUILA DEFAULT_MSG
    AQUILA_INCLUDE_DIR AQUILA_LIBRARY)
find_package_handle_standard_args(OOURA DEFAULT_MSG
    OOURA_INCLUDE_DIR OOURA_LIBRARY)

if(AQUILA_FOUND)
  set(AQUILA_LIBRARY ${AQUILA_LIBRARY})
else(AQUILA_FOUND)
  set(AQUILA_LIBRARY)
endif(AQUILA_FOUND)

if(OOURA_FOUND)
  set(OOURA_LIBRARY ${OOURA_LIBRARY})
else(OOURA_FOUND)
  set(OOURA_LIBRARY)
endif(OOURA_FOUND)

mark_as_advanced(AQUILA_INCLUDE_DIR AQUILA_LIBRARY)
mark_as_advanced(OOURA_INCLUDE_DIR OOURA_LIBRARY)
