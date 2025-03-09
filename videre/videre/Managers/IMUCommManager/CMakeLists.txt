include(rabit_add_manager)

rabit_add_manager(
    MANAGER IMUCommManager
    NAMESPACE videre
    MESSAGES_CONFIG "imucomm_mgr_config.yml"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES
        IMUCommManager.h
        IMU_DataRecord.h
        IMU_DataTypeDefs.h
        IMU_RxMessageParser.h
        IMU_TxMessageFormatter.h
        IMUCalibration.h
    SRCS
        IMUCommManager.cpp
        IMU_DataRecord.cpp
        IMU_DataTypeDefs.cpp
        IMU_RxMessageParser.cpp
        IMU_TxMessageFormatter.cpp
        IMUCalibration.cpp
    DEPENDS
        ${Boost_LIBRARIES}
        ${LOG4CXX_LIBRARIES}
        ${OpenCV_LIBS}
        rabit
        Utils
        Messages
        GeoCoordinateSystem
        ImageProcessTypes
        ImageProcessingCommonLib
    )
