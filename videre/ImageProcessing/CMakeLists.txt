project(ImageProcessingLibs)

add_subdirectory(Common)

add_subdirectory(StandardLibs)

if(WITHTESTS)
    add_subdirectory(StandardLibsUnitTests)
endif()


if(WITHCUDA)
    add_definitions(-DCUDA)
    add_subdirectory(CudaLibs)
    if(WITHTESTS)
        add_subdirectory(CudaLibsUnitTests)
    endif()
endif()



