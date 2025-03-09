project(TestCudaCommon)

#setup the NVidia NVCC toolchain
# [macro within cmake/utils.cmake]
#use_nvidia_nvcc()

find_package (Boost REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )

find_package(Log4Cxx REQUIRED)
include_directories(${LOG4CXX_INCLUDE_DIR})

file(GLOB_RECURSE ${PROJECT_NAME}_headers "${CMAKE_SOURCE_DIR}/ImageProcessing/CudaLibs/CudaCommon/*.h")
file(GLOB_RECURSE ${PROJECT_NAME}_sources "${CMAKE_SOURCE_DIR}/ImageProcessing/CudaLibs/CudaCommon/*.cpp")
file(GLOB_RECURSE ${PROJECT_NAME}_cuda_sources "${CMAKE_SOURCE_DIR}/ImageProcessing/CudaLibs/CudaCommon/*.cu")

#list(REMOVE_ITEM ${PROJECT_NAME}_sources "${CMAKE_SOURCE_DIR}/videre/main.cpp")

# This macro gets all the include directories from
# ${PROJECT_NAME}_headers and puts them into ${PROJECT_NAME}_include_dirs
get_include_dirs() # within cmake/utils.cmake
include_directories(${${PROJECT_NAME}_include_dirs})

get_common_image_processing_include_dirs()  #from utils.cmake
include_directories(${${PROJECT_NAME}_common_image_processing_include_dirs})


#Get the Unit Test headers and sources
file(GLOB_RECURSE ${PROJECT_NAME}_test_headers "*.h")
file(GLOB_RECURSE ${PROJECT_NAME}_test_sources "*.cpp")
file(GLOB_RECURSE ${PROJECT_NAME}_test_cuda_sources "*.cu")

cuda_add_executable(${PROJECT_NAME}
        ${${PROJECT_NAME}_headers}
        ${${PROJECT_NAME}_test_headers}
        ${${PROJECT_NAME}_sources}
        ${${PROJECT_NAME}_cuda_sources}
        ${${PROJECT_NAME}_test_sources}
        ${${PROJECT_NAME}_test_cuda_sources}
        )

target_link_libraries( ${PROJECT_NAME} ${Boost_LIBRARIES}
                                        ${OpenCV_LIBS}
                                        ${LOG4CXX_LIBRARIES} )


#add_test(${PROJECT_NAME} ${PROJECT_NAME})

#Make sure to add custom commands for a target after the target has been created
#copy config.ini.sample from the sample_config_files directory

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/sample_config_files/Log4CxxConfig.xml.sample"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/Log4CxxConfig.xml"
        )

#install(TARGETS testfoo DESTINATION bin)
