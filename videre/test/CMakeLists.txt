project(test)


# Setup Google Testing for unit testing
setup_gtest() # [macro within cmake/utils.cmake]
enable_testing()

add_subdirectory(test_config)
add_subdirectory(test_managers)
add_subdirectory(UnitTests)



