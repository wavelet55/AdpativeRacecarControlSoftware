include(rabit_add_manager)

rabit_add_manager(
    MANAGER ImageCaptureManager
    NAMESPACE videre
    MESSAGES_CONFIG "imgcap_mgr_config.yml"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES
        ImageCaptureManager.h
        image_capture_exception.h
        CompressedImageFileReader.h
        CSI_Camera.h
        ImageCaptureAClass.h
        ImagePlusMetaDataFileReader.h
        OpenCVWebcamImageCapture.h
        Sensoray2253ImageCapture.h
        VideoFileReader.h
        Sensoray/s2253.h
        Sensoray/s2253ioctl.h

    SRCS
        ImageCaptureManager.cpp
        CompressedImageFileReader.cpp
        CSI_Camera.cpp
        ImageCaptureAClass.cpp
        ImagePlusMetaDataFileReader.cpp
        OpenCVWebcamImageCapture.cpp
        Sensoray2253ImageCapture.cpp
        VideoFileReader.cpp
    DEPENDS
        ${Boost_LIBRARIES}
        ${LOG4CXX_LIBRARIES}
        ${OpenCV_LIBS}
        rabit
        Utils
        Messages
        ProtobufMessages
        GeoCoordinateSystem
        ImageProcessTypes
        ImageProcessingCommonLib
    )
