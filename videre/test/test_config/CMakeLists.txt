project(test_config)

file(GLOB_RECURSE ${PROJECT_NAME}_headers "../../videre/Utils/*.h")
file(GLOB_RECURSE ${PROJECT_NAME}_sources "../../videre/Utils/*.cpp")

# This macro gets all the include directories from
# ${PROJECT_NAME}_headers and puts them into ${PROJECT_NAME}_include_dirs
get_include_dirs() # within cmake/utils.cmake

file(GLOB_RECURSE test_headers "*.h")
file(GLOB_RECURSE test_sources "*.cpp")

# set(EXECUTABLE_OUTPUT_PATH "../../../")


include_directories(${${PROJECT_NAME}_include_dirs})

add_executable(${PROJECT_NAME}
        ${test_headers} ${test_sources}
        ${${PROJECT_NAME}_headers}
        ${${PROJECT_NAME}_sources})

target_link_libraries( ${PROJECT_NAME}  )

#Make sure to add custom commands for a target after the target has been created
#copy config.ini.sample from the sample_config_files directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${CMAKE_SOURCE_DIR}/sample_config_files/config.ini.sample"
                       $<TARGET_FILE_DIR:${PROJECT_NAME}>
                   )


