include(rabit_add_manager)

rabit_add_manager(
    MANAGER GPSManager
    NAMESPACE videre
    MESSAGES_CONFIG "gps_mgr_config.yml"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES
        GPSManager.h
        GPS_DataTypeDefs.h
        GPS_RxMessageParser.h
        GPSDataRecordHeader.h
        GPSDataRecord.h
    SRCS
        GPSManager.cpp
        GPS_RxMessageParser.cpp
        GPSDataRecordHeader.cpp
        GPSDataRecord.cpp
    DEPENDS
        ${Boost_LIBRARIES}
        ${LOG4CXX_LIBRARIES}
        ${OpenCV_LIBS}
        Utils
        Messages
        ImageProcessingCommonLib
        GeoCoordinateSystem
        NemaTodeGpsParser
    )
