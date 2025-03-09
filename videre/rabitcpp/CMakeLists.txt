#************************************************************************************
# 4/30/2016
# Randy Direen
#
# Build system for Rabit
#
#************************************************************************************
cmake_minimum_required (VERSION 3.10 FATAL_ERROR)
project (rabit_dev VERSION 1.0.1 LANGUAGES CXX)

#************************************************************************************
# Setup Versioning
#************************************************************************************

# The version number.
set(rabit_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")


# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/rabitConfig.h.in"
  "${PROJECT_BINARY_DIR}/rabitConfig.h"
  )
  

option(WITHTESTS "Compile code with unit tests and integration tests" OFF)
option(WITHEXAMPLES "Compile code with examples" OFF)
option(WITHCOVERAGE "Compile code with code coverage (for online CI)" OFF)

#set(WITHTESTS true)
#set(WITHEXAMPLES true)

if(WITHCOVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0") # debug, no optimization
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage") # enabling coverage
endif()
#************************************************************************************
# Include some macros to make CMakeLists files cleaner
#************************************************************************************
include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Setup Google Testing for unit testing
if(WITHTESTS)
  setup_gtest() # [macro within cmake/utils.cmake]
  enable_testing()
endif()

#************************************************************************************
# The subdirectories
#************************************************************************************
add_subdirectory(rabit) 

if(WITHEXAMPLES)
  add_subdirectory(examples)
endif()

if(WITHTESTS)
  add_subdirectory(test)
endif()




