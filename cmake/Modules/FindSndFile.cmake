# - Find sndfile
# Find the native sndfile includes and libraries
#
#  SNDFILE_INCLUDE_DIR - where to find sndfile.h, etc.
#  SNDFILE_LIBRARIES   - List of libraries when using libsndfile.
#  SNDFILE_FOUND       - True if libsndfile found.

FIND_PATH(SNDFILE_INCLUDE_DIR NAMES sndfile.h PATHS ${CMAKE_SOURCE_DIR}/libs/libsndfile/src/)

FIND_LIBRARY(SNDFILE_LIBRARY NAMES libsndfile.a PATHS ${CMAKE_SOURCE_DIR}/libs/libsndfile/src/.libs/)

MESSAGE( STATUS "SNDFILE_INCLUDE_DIR = \"${SNDFILE_INCLUDE_DIR}\"" )
MESSAGE( STATUS "SNDFILE_LIBRARY = \"${SNDFILE_LIBRARY}\"" )


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SNDFILE DEFAULT_MSG
    SNDFILE_INCLUDE_DIR SNDFILE_LIBRARY)

if(SNDFILE_FOUND)
  set(SNDFILE_LIBRARIES ${SNDFILE_LIBRARY})
else(SNDFILE_FOUND)
  set(SNDFILE_LIBRARIES)
endif(SNDFILE_FOUND)

mark_as_advanced(SNDFILE_INCLUDE_DIR SNDFILE_LIBRARY)
