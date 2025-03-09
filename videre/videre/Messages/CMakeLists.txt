project(Messages)

message("[videre] Setting up Messages OBJECT library build")

if(WITHCUDA)
    add_definitions(-DCUDA)
endif()

set(SRCS_H 
        ${CMAKE_CURRENT_SOURCE_DIR}/all_manager_message.h
        ${CMAKE_CURRENT_SOURCE_DIR}/AccelerometerGyroMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraCalCommandMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraCalStatusMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraOrientationMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraParametersSetupMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CompressedImageMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/DceEPASteeringStatusMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/FeatureMatchProcCtrlMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GPSFixMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/HeadOrientationCalDataMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/HeadOrientationControlMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/HeadOrientationMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/HeadTrackingControlMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/HeadTrackingOrientationMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/image_plus_metadata_message.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ImageCaptureControlMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ImageCaptureStatusMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ImageProcessControlMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ImageLoggingControlMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ImageProcTargetInfoResultsMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/IMUCommandResponseMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/KarTechLinearActuatorParamsMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/KarTechLinearActuatorSetupMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/LinearActuatorPositionCtrlMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/PlaybackControlMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/QuaternionMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/RemoteControlInputMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ResetOrientationStateMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/SipnPuffControlMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/SipnPuffMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/SteeringTorqueCtrlMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/StreamRecordImagesControlMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/sysinfo_dynamic_message.h
        ${CMAKE_CURRENT_SOURCE_DIR}/sysinfo_static_message.h
        ${CMAKE_CURRENT_SOURCE_DIR}/TargetParametersMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/telemetry_message.h
        ${CMAKE_CURRENT_SOURCE_DIR}/VehicleControlParametersMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/VehicleInertialStatesMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/VehicleSwitchInputMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/video_control_message.h
        ${CMAKE_CURRENT_SOURCE_DIR}/video_process_message.h
        ${CMAKE_CURRENT_SOURCE_DIR}/VidereSystemControlMessage.h
        ${CMAKE_CURRENT_SOURCE_DIR}/MsgWrapper.h
        )

set(SRCS_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraOrientationMessage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CameraParametersSetupMessage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CompressedImageMessage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/FeatureMatchProcCtrlMessage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GPSFixMessage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/image_plus_metadata_message.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ImageCaptureContolMessage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ImageCaptureStatusMessage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ImageProcTargetInfoResultsMessage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TargetParametersMessage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/VehicleInertialStatesMessage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/MsgWrapper.cpp
        )


add_library(${PROJECT_NAME} STATIC ${SRCS_CPP} ${SRCS_H})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(WITHCUDA)
    target_link_libraries(${PROJECT_NAME} PRIVATE
            ${Boost_LIBRARIES}
            ${OpenCV_LIBS}
            rabit Utils
            fastrtps fastcdr
            GeoCoordinateSystem
            ImageProcessTypes
            ImageProcessingCommonLib
            CudaImageProcessingLibs)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE
            ${Boost_LIBRARIES}
            ${OpenCV_LIBS}
            rabit Utils
            ProtobufMessages
            fastrtps fastcdr
            GeoCoordinateSystem
            NemaTodeGpsParser
            ImageProcessTypes
            ImageProcessingCommonLib)

endif()