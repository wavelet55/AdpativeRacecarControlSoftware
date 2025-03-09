
# ----------------------------------------------------------------------------
# run this to setup the NVidia NVCC compiler for CUDA Software
# ----------------------------------------------------------------------------
macro(cuda_nvcc_options_setup)
    find_package(CUDA QUIET REQUIRED)

    INCLUDE(FindCUDA)
    INCLUDE_DIRECTORIES(/usr/local/cuda/include)

    #set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)

    #set(BUILD_SHARED_LIBS ON)

    #set(CUDA_SEPARABLE_COMPILATION ON)

    #Block -std-c++0x from getting added to nvcc
    #set(CUDA_PROPAGATE_HOST_FLAGS OFF)

    LIST(APPEND CUDA_NVCC_FLAGS
            --compiler-options
            -fno-strict-aliasing
            -lineinfo
            -use_fast_math
            -std=c++11
            )

    #Choose the version of Cuda you need.
        if(WITHTK1)
            LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
            message("Using NVidia TK1 Compiler Options arch=compute_20,code=sm_20")
        elseif(WITHTX1)
            #LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_53,code=sm_53)
            LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
            message("Using NVidia TX1 Compiler Options arch=compute_53,code=sm_53")
        elseif(WITHTX2)  #
            LIST(APPEND CUDA_NVCC_FLAGS -arch=sm_62 -gencode=arch=compute_62,code=sm_62)
            message("Using NVidia TX2 Compiler Options arch=compute_62,code=sm_62 ")
        else()  #Default is XAVIER
            LIST(APPEND CUDA_NVCC_FLAGS -arch=sm_72 -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_62,code=sm_62)
            message("Using NVidia XAVIER Compiler Options arch=compute_72,code=sm_72 ")
        endif()

    #set(CUDA_LIBRARIES /usr/local/cuda/lib/libcudart.so)
    message("Cuda NVCC Flags are: ${CUDA_NVCC_FLAGS}")

endmacro(cuda_nvcc_options_setup)

# ----------------------------------------------------------------------------
# run this to setup the NVidia ARM Cross compiler
# ----------------------------------------------------------------------------
macro(use_nvidia_arm_crosscompiler)


endmacro(use_nvidia_arm_crosscompiler)

# ----------------------------------------------------------------------------
# run this to make sure we are using C++11
# ----------------------------------------------------------------------------
macro(use_cxx11)
    if (CMAKE_VERSION VERSION_LESS "3.1")
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")
        endif ()
        # Might need to add something in here if we are using other operating systems.
    else ()
        set(CMAKE_CXX_STANDARD 11)
    endif ()
endmacro(use_cxx11)

# ----------------------------------------------------------------------------
# run this to find and include Rabit Include Directories
# ----------------------------------------------------------------------------
macro(get_rabit_include_dirs)
    set(${PROJECT_NAME}_rabit_include_dirs "")
    file(GLOB_RECURSE ${PROJECT_NAME}_rabitheaders "${CMAKE_SOURCE_DIR}/rabitcpp/rabit/*.h")

    foreach (_headerFile ${${PROJECT_NAME}_rabitheaders})
        get_filename_component(_dir ${_headerFile} PATH)
        list(APPEND ${PROJECT_NAME}_rabit_include_dirs ${_dir})
    endforeach ()
    list(REMOVE_DUPLICATES ${PROJECT_NAME}_rabit_include_dirs)
endmacro(get_rabit_include_dirs)


# ----------------------------------------------------------------------------
# run this to find include directories
# ----------------------------------------------------------------------------
macro(get_include_dirs)
    set(${PROJECT_NAME}_include_dirs "")
    foreach (_headerFile ${${PROJECT_NAME}_headers})
        get_filename_component(_dir ${_headerFile} PATH)
        list(APPEND ${PROJECT_NAME}_include_dirs ${_dir})
    endforeach ()
    list(REMOVE_DUPLICATES ${PROJECT_NAME}_include_dirs)
endmacro(get_include_dirs)

# ----------------------------------------------------------------------------
# run this to find and include Image Processing Common Include Directories
# ----------------------------------------------------------------------------
macro(get_common_image_processing_include_dirs)
    set(${PROJECT_NAME}_common_image_processing_include_dirs "")
    file(GLOB_RECURSE ${PROJECT_NAME}_commonimgprocheaders "${CMAKE_SOURCE_DIR}/ImageProcessing/Common/*.h")

    foreach (_headerFile ${${PROJECT_NAME}_commonimgprocheaders})
        get_filename_component(_dir ${_headerFile} PATH)
        list(APPEND ${PROJECT_NAME}_common_image_processing_include_dirs ${_dir})
    endforeach ()
    list(REMOVE_DUPLICATES ${PROJECT_NAME}_common_image_processing_include_dirs)
endmacro(get_common_image_processing_include_dirs)


# ----------------------------------------------------------------------------
# run this to find and include Standard Image Processing Include Directories
# ----------------------------------------------------------------------------
macro(get_std_image_processing_include_dirs)
    set(${PROJECT_NAME}_std_image_processing_include_dirs "")
    file(GLOB_RECURSE ${PROJECT_NAME}_stdimgprocheaders "${CMAKE_SOURCE_DIR}/ImageProcessing/StandardLibs/*.h")

    foreach (_headerFile ${${PROJECT_NAME}_stdimgprocheaders})
        get_filename_component(_dir ${_headerFile} PATH)
        list(APPEND ${PROJECT_NAME}_std_image_processing_include_dirs ${_dir})
    endforeach ()
    list(REMOVE_DUPLICATES ${PROJECT_NAME}_std_image_processing_include_dirs)
endmacro(get_std_image_processing_include_dirs)


# ----------------------------------------------------------------------------
# run this to find and include Standard Image Processing Include Directories
# ----------------------------------------------------------------------------
macro(get_cuda_image_processing_include_dirs)
    set(${PROJECT_NAME}_cuda_image_processing_include_dirs "")
    file(GLOB_RECURSE ${PROJECT_NAME}_cudaimgprocheaders "${CMAKE_SOURCE_DIR}/ImageProcessing/CudaLibs/*.h")

    foreach (_headerFile ${${PROJECT_NAME}_cudaimgprocheaders})
        get_filename_component(_dir ${_headerFile} PATH)
        list(APPEND ${PROJECT_NAME}_cuda_image_processing_include_dirs ${_dir})
    endforeach ()
    list(REMOVE_DUPLICATES ${PROJECT_NAME}_cuda_image_processing_include_dirs)
endmacro(get_cuda_image_processing_include_dirs)


# ----------------------------------------------------------------------------
# run this to setup gtest
# ----------------------------------------------------------------------------
macro(setup_gtest)
    # We need thread support
    find_package(Threads REQUIRED)

    # Enable ExternalProject CMake module
    include(ExternalProject)

    # Download and install GoogleMock
    ExternalProject_Add(
            gmock
            #URL https://github.com/google/googletest/gmock-1.7.0.zip
            URL http://pkgs.fedoraproject.org/repo/pkgs/gmock/gmock-1.7.0.zip/073b984d8798ea1594f5e44d85b20d66/gmock-1.7.0.zip
            PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gmock
            # Disable install step
            INSTALL_COMMAND ""
    )

    # Create a libgmock target to be used as a dependency by test programs
    add_library(libgmock IMPORTED STATIC GLOBAL)
    add_dependencies(libgmock gmock)

    # Set gmock properties
    ExternalProject_Get_Property(gmock source_dir binary_dir)
    set_target_properties(libgmock PROPERTIES
            "IMPORTED_LOCATION" "${binary_dir}/libgmock.a"
            "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
            )
    include_directories("${source_dir}/include")
    include_directories(${source_dir}/gtest/include)

endmacro(setup_gtest)
