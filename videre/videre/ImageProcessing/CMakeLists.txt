project(ImageProcessTypes)

message("[videre] Setting up Image Process Types OBJECT library build")

if(WITHCUDA)
    add_definitions(-DCUDA)
endif()

set(RABIT_MSG_INCLUDE_DIR ${CMAKE_BINARY_DIR}/rabit_message_headers)
message("[ImageProcessing] Rabit Message Dir: ${RABIT_MSG_INCLUDE_DIR}")

set(SRCS_H 
        ${CMAKE_CURRENT_SOURCE_DIR}/CheckerBdTargetDetector.h
        ${CMAKE_CURRENT_SOURCE_DIR}/TargetDetector.h
        ${CMAKE_CURRENT_SOURCE_DIR}/TargetDetectorProcessControl.h
        ${CMAKE_CURRENT_SOURCE_DIR}/BlobTargetDetectionProcess/BlobTargetDetector.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraCalibration/CameraCalibrationProcessControl.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraCalibration/CameraCalibrationWith2DObjects.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraCalibration/CameraCalReaderWriter.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraCalibration/popt_pp.h
        ${CMAKE_CURRENT_SOURCE_DIR}/RecorderPlayer/ImagePlusMetadataFileHeaders.h
        ${CMAKE_CURRENT_SOURCE_DIR}/RecorderPlayer/ImagePlusMetadataReader.h
        ${CMAKE_CURRENT_SOURCE_DIR}/RecorderPlayer/ImagePlusMetadataRecorder.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/CameraOrientationValidation.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/ImageMetadataWriter.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/ImagePixelLocationToRealWorldLocation.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/JpgFileHandling.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/HeadTrackingCalParamsReaderWriter.h
        )

set(SRCS_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/CheckerBdTargetDetector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TargetDetector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TargetDetectorProcessControl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/BlobTargetDetectionProcess/BlobTargetDetector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraCalibration/CameraCalibrationProcessControl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraCalibration/CameraCalibrationWith2DObjects.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraCalibration/CameraCalReaderWriter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RecorderPlayer/ImagePlusMetadataFileHeaders.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RecorderPlayer/ImagePlusMetadataReader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/RecorderPlayer/ImagePlusMetadataRecorder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/CameraOrientationValidation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/ImageMetadataWriter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/ImagePixelLocationToRealWorldLocation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/JpgFileHandling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/HeadTrackingCalParamsReaderWriter.cpp
        )


add_library(${PROJECT_NAME} STATIC ${SRCS_CPP} ${SRCS_H})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )
target_include_directories(${PROJECT_NAME} PUBLIC ${RABIT_MSG_INCLUDE_DIR})
if(WITHCUDA)
    message("[videre] Generating Image Process Types OBJECT Lib with Cuda Processing:")
    target_link_libraries(${PROJECT_NAME} PRIVATE rabit Utils
            GeoCoordinateSystem
            Utils
            rabit_msgs fastrtps fastcdr
            Messages
            ImageProcessingCommonLib
            StandardImageProcessingLibs
            CudaImageProcessingLibs
            ${OpenCV_LIBS}
            )
else()
    message("[videre] Generating Image Process Types OBJECT Lib without Nvidia Cuda Processing:")
    target_link_libraries(${PROJECT_NAME} PRIVATE rabit Utils
            GeoCoordinateSystem
            Utils
            rabit_msgs fastrtps fastcdr
            Messages
            ImageProcessingCommonLib
            StandardImageProcessingLibs
            ${OpenCV_LIBS}
            )
endif()
