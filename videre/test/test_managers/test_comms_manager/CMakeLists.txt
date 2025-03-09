project(test_comms_manager)

find_package (Boost REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )
find_package( ZMQ REQUIRED )
include_directories(${ZMQ_INCLUDE_DIR})
find_package( Protobuf REQUIRED )
include_directories(${PROTOBUF_INCLUDE_DIRS})
find_package( Rabit REQUIRED )
include_directories(${RABIT_INCLUDE_DIR})
find_package(Log4Cxx REQUIRED)
include_directories(${LOG4CXX_INCLUDE_DIR})


file(GLOB_RECURSE ${PROJECT_NAME}_headers "${CMAKE_SOURCE_DIR}/videre/*.h")
file(GLOB_RECURSE ${PROJECT_NAME}_sources "${CMAKE_SOURCE_DIR}/videre/*.cpp")
list(REMOVE_ITEM ${PROJECT_NAME}_sources "${CMAKE_SOURCE_DIR}/videre/main.cpp")

get_include_dirs() # see CMAKE utils.cmake
include_directories(${${PROJECT_NAME}_include_dirs})

file(GLOB_RECURSE test_headers "*.h")
file(GLOB_RECURSE test_sources "*.cpp")

add_executable(${PROJECT_NAME} ${test_headers} ${${PROJECT_NAME}_headers}
                                ${test_sources} ${${PROJECT_NAME}_sources} )

target_link_libraries( ${PROJECT_NAME} ${Boost_LIBRARIES}
                                       ${OpenCV_LIBS}
                                       ${ZMQ_LIBRARIES}
                                       ${PROTOBUF_LIBRARIES}
                                        ${LOG4CXX_LIBRARIES}
                                       ${RABIT_LIBRARIES} )

#Make sure to add custom commands for a target after the target has been created
#copy config.ini.sample from the sample_config_files directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${CMAKE_SOURCE_DIR}/sample_config_files/config.ini.sample"
                       $<TARGET_FILE_DIR:${PROJECT_NAME}>
                   )
