#cmake_minimum_required(VERSION 3.14)
project(Utils)

message("[videre] Building Utils OBJECT Library")

find_package (Boost REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )

find_package( ZMQ REQUIRED )
include_directories(${ZMQ_INCLUDE_DIR})

set(SRCS_H 
        ${CMAKE_CURRENT_SOURCE_DIR}/FileUtils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ByteArrayReaderWriterVidere.h
        ${CMAKE_CURRENT_SOURCE_DIR}/LockFreeQueue.h
        ${CMAKE_CURRENT_SOURCE_DIR}/HostSystemInfo.h
        ${CMAKE_CURRENT_SOURCE_DIR}/RS232Comm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/SerialCommMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/config_data.h
        ${CMAKE_CURRENT_SOURCE_DIR}/config_parse_exception.h
        ${CMAKE_CURRENT_SOURCE_DIR}/global_defines.h
        ${CMAKE_CURRENT_SOURCE_DIR}/logger.h
        ${CMAKE_CURRENT_SOURCE_DIR}/timing.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CubicSplineProcessor.h
        ${CMAKE_CURRENT_SOURCE_DIR}/DataLogReader.h
        ${CMAKE_CURRENT_SOURCE_DIR}/DataRecorder.h
        ${CMAKE_CURRENT_SOURCE_DIR}/DataRecorderAbstractRecord.h
        ${CMAKE_CURRENT_SOURCE_DIR}/DataRecorderStdHeader.h
        )

set(SRCS_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/FileUtils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ByteArrayReaderWriterVidere.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/HostSystemInfo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RS232Comm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/SerialCommMessage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/config_data.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/global_defines.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CubicSplineProcessor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DataLogReader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DataRecorder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DataRecorderAbstractRecord.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DataRecorderStdHeader.cpp
        )


add_library(${PROJECT_NAME} STATIC ${SRCS_CPP} ${SRCS_H})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE 
                                        ${Boost_LIBRARIES}
                                        ${OpenCV_LIBS}
                                        rabit
                                        Messages
                                        GeoCoordinateSystem
                                        ImageProcessingCommonLib)
