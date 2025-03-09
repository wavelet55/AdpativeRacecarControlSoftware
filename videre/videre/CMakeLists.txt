project(videre)

message("CMAKE Project: ${PROJECT_NAME}")

execute_process(
        COMMAND git describe --always --tags
        OUTPUT_VARIABLE GIT_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
message("[videre]  Videre Version Git Tag: ${GIT_TAG}")


#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${TOPLEVEL_DIR}/videre/cmake/Modules/")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
#message("CMAKE MODULE PATH: ${CMAKE_MODULE_PATH}")

#==============================================================================
#            Find and included 3rd party libraries
#==============================================================================

find_package (Boost REQUIRED)
find_package(Boost COMPONENTS system filesystem thread REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )
find_package( ZMQ REQUIRED )
include_directories(${ZMQ_INCLUDE_DIR})
find_package( Protobuf REQUIRED )
include_directories(${PROTOBUF_INCLUDE_DIRS})
#find_package( Rabit REQUIRED )
#include_directories(${RABIT_INCLUDE_DIR})
find_package(Log4Cxx REQUIRED)
include_directories(${LOG4CXX_INCLUDE_DIR})

#Fast DDS/RTPS
#https://www.eprosima.com/index.php/products-all/eprosima-fast-dds
find_package( fastrtps REQUIRED )
find_package( fastcdr REQUIRED )

#==============================================================================
#            Setup for building libraries local to this project
#==============================================================================
add_subdirectory( Utils )     # General Utilities
add_subdirectory( Messages )  # Custom Rabit Messages
add_subdirectory( ProtobufMessages ) #Auto Generated Protobuf Messages
add_subdirectory( RabitMsgs ) # Auto Generated Rabit Messages
add_subdirectory( GeoCoordinateSystem )
add_subdirectory( NemaTodeGpsParser )
add_subdirectory( ImageProcessing )

message("[videre] Messages Used In This Project:")
foreach(msg ${RABIT_SYSTEM_MESSAGES})
    message("[videre]      - ${msg}")
endforeach()

set(RABIT_CONFIGURATION "product" CACHE STRING "Select rabit configuration file")
#set(RABIT_CONFIGURATION "dtx_imu" CACHE STRING "Select rabit configuration file")
message("[videre] Manager Configuration: ${RABIT_CONFIGURATION}")

#==============================================================================
#            Configure build for MAIN executable
#==============================================================================
include(rabit_executable)

message("[videre] Create the Rabit Executable:")

if(WITHCUDA)
    message("[videre] Build Videre with CUDA:")
    add_definitions(-DCUDA)
    rabit_executable(
            TARGET_NAME ${PROJECT_NAME}
            RABIT_CONFIGURATION ${CMAKE_CURRENT_SOURCE_DIR}/Configurations/${RABIT_CONFIGURATION}_configuration.cmake
            MANAGER_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/Managers
            MAIN_TEMPLATE main.cpp.em
            MAIN_TEMPLATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
            INCLUDE_DIRECTORIES
                ${Boost_INCLUDE_DIRS}
                ${LOG4CXX_INCLUDE_DIR}
            LIBRARIES
                ${Boost_LIBRARIES}
                ${ZMQ_LIBRARIES}
                ${LOG4CXX_LIBRARIES}
                ${PROTOBUF_LIBRARIES}
                Utils
                Messages
                ProtobufMessages
                fastrtps
                fastcdr
                GeoCoordinateSystem
                NemaTodeGpsParser
                ImageProcessTypes
                ImageProcessingCommonLib
                StandardImageProcessingLibs
                CudaImageProcessingLibs
            )
else()
    message("[videre] Build Videre NO NVidia CUDA Processing:")
    rabit_executable(
            TARGET_NAME ${PROJECT_NAME}
            RABIT_CONFIGURATION ${CMAKE_CURRENT_SOURCE_DIR}/Configurations/${RABIT_CONFIGURATION}_configuration.cmake
            MANAGER_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/Managers
            MAIN_TEMPLATE main.cpp.em
            MAIN_TEMPLATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
            INCLUDE_DIRECTORIES
            ${Boost_INCLUDE_DIRS}
            ${LOG4CXX_INCLUDE_DIR}
            LIBRARIES
            ${Boost_LIBRARIES}
            ${ZMQ_LIBRARIES}
            ${LOG4CXX_LIBRARIES}
            ${PROTOBUF_LIBRARIES}
            Utils
            Messages
            ProtobufMessages
            fastrtps
            fastcdr
            GeoCoordinateSystem
            NemaTodeGpsParser
            ImageProcessTypes
            ImageProcessingCommonLib
            StandardImageProcessingLibs
        )
endif()

message("[videre] Done Creating the Rabit Executable:")

#copy files to output
set(VIDERE_CONFIG_FILE_IN "VidereConfig.ini.in")
set(VIDERE_CONFIG_FILE "VidereConfig.ini")

set(LOG_CONFIG_FILE_IN "Log4CxxConfig.xml.in")
set(LOG_CONFIG_FILE "Log4CxxConfig.xml")

set(TAIL_FILE_IN "tail_logs.py.in")
set(TAIL_FILE "tail_logs.py")

#Copy Over Protobuf messages if they have been updated.
#Make sure to add custom commands for a target after the target has been created
#copy config.ini.sample from the sample_config_files directory
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/../protobuf_messages/gen_source/cpp/vision_messages.pb.h"
        "${CMAKE_SOURCE_DIR}/videre/ProtobufMessages/vision_messages.pb.h"
        )


#Make sure to add custom commands for a target after the target has been created
#copy config.ini.sample from the sample_config_files directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${CMAKE_SOURCE_DIR}/sample_config_files/VidereConfig.ini.sample"
                       "$<TARGET_FILE_DIR:${PROJECT_NAME}>/VidereConfig.ini"
                   )

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/sample_config_files/Log4CxxConfig.xml.sample"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/Log4CxxConfig.xml"
        )

add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${TAIL_FILE}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/Tools/${TAIL_FILE_IN}"
        "${CMAKE_CURRENT_BINARY_DIR}/${TAIL_FILE}"
        DEPENDS "${CMAKE_SOURCE_DIR}/Tools/${TAIL_FILE_IN}"
        COMMENT "[videre] Generating ${TAIL_FILE}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
)
add_custom_target(generate_python_tail ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${TAIL_FILE}")

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/Tools/tail_logs.py.in"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/tail_logs.py"
        )
