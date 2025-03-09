include(rabit_add_manager)

rabit_add_manager(
    MANAGER StreamRecordManager
    NAMESPACE videre
    MESSAGES_CONFIG "streamrcd_mgr_config.yml"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES
        StreamRecordManager.h
        vision_record_play.h
    SRCS
        StreamRecordManager.cpp
        vision_record_play.cpp
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
    )
