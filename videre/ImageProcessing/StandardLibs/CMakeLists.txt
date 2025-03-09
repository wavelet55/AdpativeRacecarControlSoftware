project(StandardImageProcessingLibs)

#add_subdirectory(StdCommon)
#add_subdirectory(TargetDetection)
#add_subdirectory(GPSDeniedImageProcessing)

message("CMAKE Project: ${PROJECT_NAME}")

find_package (Boost REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )

find_package(Log4Cxx REQUIRED)
include_directories(${LOG4CXX_INCLUDE_DIR})

add_subdirectory( GPSDeniedImageProcessing )

set(SRCS_H
        ${CMAKE_CURRENT_SOURCE_DIR}/StdCommon/ImageFeatureMatcher.h
        ${CMAKE_CURRENT_SOURCE_DIR}/StdCommon/ImageKeypointsAndFeatures.h
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/Blob.h
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/BlobExtraction.h
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/BlobLibraryConfiguration.h
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/BlobResult.h
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/StdBlobTargetDetector.h
        ${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetectorOpenCVSimple/StdBlobTargetDetectorOpenCVSimple.h
        ${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/CheckerBoardTargetDetector/CheckerBoardTargetDetector.h
        )

set(SRCS_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/StdCommon/ImageFeatureMatcher.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/StdCommon/ImageKeypointsAndFeatures.cpp
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/Blob.cpp
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/BlobExtraction.cpp
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/BlobResult.cpp
        #${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetector/StdBlobTargetDetector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/BlobTargetDetectorOpenCVSimple/StdBlobTargetDetectorOpenCVSimple.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TargetDetection/CheckerBoardTargetDetector/CheckerBoardTargetDetector.cpp
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

