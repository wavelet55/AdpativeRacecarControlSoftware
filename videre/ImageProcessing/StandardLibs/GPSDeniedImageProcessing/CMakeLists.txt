project(StandardGPSDeniedImageProcessing)

message("CMAKE Project: ${PROJECT_NAME}")

find_package (Boost REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )

find_package(Log4Cxx REQUIRED)
include_directories(${LOG4CXX_INCLUDE_DIR})

set(SRCS_H
        ${CMAKE_CURRENT_SOURCE_DIR}/GPSDeniedImageProcessor.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ImageAlignment.h
        )

set(SRCS_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/GPSDeniedImageProcessor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ImageAlignment.cpp
        )


add_library(${PROJECT_NAME} STATIC ${SRCS_CPP} ${SRCS_H})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        ${LOG4CXX_LIBRARIES}
        ImageProcessingCommonLib
        )


#Make sure to add custom commands for a target after the target has been created
#copy config.ini.sample from the sample_config_files directory
#Make sure to add custom commands for a target after the target has been created
#copy config.ini.sample from the sample_config_files directory

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/sample_config_files/Log4CxxConfig.xml.sample"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/Log4CxxConfig.xml"
        )