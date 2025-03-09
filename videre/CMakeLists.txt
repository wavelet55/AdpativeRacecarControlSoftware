#************************************************************************************
# 8/23/2021
# Randy Direen Ph.D. (rdireen@direentech.com)
# Harry Direen Ph.D. P.E. (hdireen@direentech.com)
#
# Build system for Videre
#
#************************************************************************************
cmake_minimum_required (VERSION 3.10 FATAL_ERROR)
project (videre_racecar LANGUAGES CXX)

#set(ProjectVidere ${CMAKE_CURRENT_SOURCE_DIR})

#************************************************************************************
# Setup Versioning
#************************************************************************************
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Make sure we are using the C++11 standard
#use_cxx11() # [macro within cmake/utils.cmake]

#Ensure we are compiling with pthreads
LIST(APPEND CMAKE_CXX_FLAGS -pthread)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/videreConfig.h.in"
  "${PROJECT_BINARY_DIR}/videreConfig.h"
  )


#Set -D XCTK1
option(XCBD "Cross Compile for specific NVidia Board: [OFF, TK1, TX1, TX2]" OFF)
if(XCBD)
    message("Cross Compile for NVidia Board: ${XCBD}")
endif()

option(BDIPADDR "IP Address for the NVidia Board" OFF)
if(BDIPADDR)
    message("NVidia Board IP Addr: ${BDIPADDR}")
endif()

#Set -DWITHCUDA inorder to compile the NVidia CUDA based GPGPU
#libraries and to cause Videre to look for and include these
#libraries.
option(WITHCUDA "Compile NVidia CUDA GPGPU Libraries" OFF)
#set(WITHCUDA TRUE)

option(WITHTK1 "Nvidia GPU Board: TK1" OFF)
option(WITHTX1 "Nvidia GPU Board: TX1" OFF)
option(WITHTX2 "Nvidia GPU Board: TX2" OFF)
option(WITHXAVIER "Nvidia GPU Board: XAVIER" OFF)

#Spit out the Compiler Flags for reference.
if(WITHCUDA)
    message("Using NVidia CUDA Libraries")
    if(WITHTK1)
        message("Using NVidia TK1 Compiler Options")
    elseif(WITHTX1)
        message("Using NVidia TX1 Compiler Options")
    elseif(WITHTX2)
        message("Using NVidia TX2 Compiler Options")
    else()
        message("Using NVidia XAVIER Compiler Options")
    endif()
else()
    message("NVidia CUDA is OFF")
endif()

option(WITHTESTS "Compile code with unit tests and integration tests" OFF)
option(WITHEXAMPLES "Compile code with examples" OFF)
option(WITHCOVERAGE "Compile code with code coverage (for online CI)" OFF)

#set(WITHTESTS TRUE)
#set(WITHEXAMPLES TRUE)

if(WITHCOVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0") # debug, no optimization
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage") # enabling coverage
endif()


#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
message("CMAKE MODULE PATH: ${CMAKE_MODULE_PATH}")

# add the binary tree to the search path for include files
# so that we will find rabitConfig.h
include_directories("${PROJECT_BINARY_DIR}")

#************************************************************************************
# Include some macros to make CMakeLists files cleaner
#************************************************************************************
include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)


#************************************************************************************
#  Find / Establish the Python Interpreter
#************************************************************************************

find_package(PythonInterp 3 REQUIRED)
# We have a custom error message to tell users how to install python3.
if (NOT PYTHONINTERP_FOUND)
    message(FATAL_ERROR "Python 3 not found. Please install Python 3:\n"
            "    Ubuntu: sudo apt install python3 python3-dev python3-pip\n"
            "    macOS: brew install python")
endif()

message("Python_FOUND:${Python_FOUND}")
message("Python_VERSION:${Python_VERSION}")
#message("Python_Development_FOUND:${Python_Development_FOUND}")
#message("Python_LIBRARIES:${Python_LIBRARIES}")


#************************************************************************************
# The subdirectories
#************************************************************************************

get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)
get_filename_component(TOPLEVEL_DIR ${PARENT_DIR} DIRECTORY)

message("[videre] PROJECT_BINARY_DIR is ${PROJECT_BINARY_DIR}")
message("[videre] PROJECT_SOURCE_DIR is ${PROJECT_SOURCE_DIR}")
message("[videre] PARENT_DIR is ${PARENT_DIR}")
message("[videre] TOPLEVEL_DIR is ${TOPLEVEL_DIR}")


set(RBX_TOOLS ${CMAKE_CURRENT_SOURCE_DIR}/rabitcpp/rbx_tools)
set(CMAKE_MODULE_PATH ${RBX_TOOLS} ${CMAKE_MODULE_PATH})

add_subdirectory(rabitcpp)

#Build Image Processing Libararies first so that they are
#available for videre.
add_subdirectory(ImageProcessing)

add_subdirectory(videre) 

if(WITHTESTS)
  add_subdirectory(test)
endif()

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
message("CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")
message("CUDA is ${WITHCUDA}")
message("TK1 is ${WITHTK1}")
message("TX1 is ${WITHTX1}")
message("TX2 is ${WITHTX2}")
message("XAVIER is ${WITHXAVIER}")
