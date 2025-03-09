include(rabit_add_manager)

if(WITHCUDA)
     add_definitions(-DCUDA)
     rabit_add_manager(
         MANAGER VisionProcessManager
         NAMESPACE videre
         MESSAGES_CONFIG "visionproc_mgr_config.yml"
         WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
         WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
         INCLUDES
             CameraCalibrationProcessor.h
             VisionProcessManager.h
             FeatureMatchingProcessor.h
             GPSDeniedVisionProcessor.h
             HeadOrientationCalProcessor.h
             HeadTrackingProcessor.h
             NullVisionProcessor.h
             StdTargetDectectorVisionProcessor.h
             TrackHeadDataRecord.h
             VisionProcessorAbstract.h
             VisionProcessResources.h
         SRCS
             CameraCalibrationProcessor.cpp
             VisionProcessManager.cpp
             FeatureMatchingProcessor.cpp
             GPSDeniedVisionProcessor.cpp
             HeadOrientationCalProcessor.cpp
             HeadTrackingProcessor.cpp
             NullVisionProcessor.cpp
             StdTargetDectectorVisionProcessor.cpp
             TrackHeadDataRecord.cpp
             VisionProcessorAbstract.cpp
             VisionProcessResources.cpp
         DEPENDS
             ${Boost_LIBRARIES}
             ${LOG4CXX_LIBRARIES}
             ${OpenCV_LIBS}
             Utils
             Messages
             ProtobufMessages
             GeoCoordinateSystem
             ImageProcessTypes
             ImageProcessingCommonLib
             StandardImageProcessingLibs
             CudaImageProcessingLibs
          )
else()
     rabit_add_manager(
             MANAGER VisionProcessManager
             NAMESPACE videre
             MESSAGES_CONFIG "visionproc_mgr_config.yml"
             WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
             WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
             INCLUDES
                  CameraCalibrationProcessor.h
                  VisionProcessManager.h
                  FeatureMatchingProcessor.h
                  GPSDeniedVisionProcessor.h
                  NullVisionProcessor.h
                  StdTargetDectectorVisionProcessor.h
                  VisionProcessorAbstract.h
                  VisionProcessResources.h
             SRCS
                  CameraCalibrationProcessor.cpp
                  VisionProcessManager.cpp
                  FeatureMatchingProcessor.cpp
                  GPSDeniedVisionProcessor.cpp
                  NullVisionProcessor.cpp
                  StdTargetDectectorVisionProcessor.cpp
                  VisionProcessorAbstract.cpp
                  VisionProcessResources.cpp
             DEPENDS
                  ${Boost_LIBRARIES}
                  ${LOG4CXX_LIBRARIES}
                  ${OpenCV_LIBS}
                  Utils
                  Messages
                  ProtobufMessages
                  GeoCoordinateSystem
                  ImageProcessTypes
                  ImageProcessingCommonLib
                  StandardImageProcessingLibs
     )
endif()