include(rabit_add_manager)

rabit_add_manager(
    MANAGER SipnPuffManager
    NAMESPACE videre
    MESSAGES_CONFIG "sipnpuff_mgr_config.yml"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES
        SipnPuffManager.h
        SipnPuffDataRecord.h
    SRCS
        SipnPuffManager.cpp
        SipnPuffDataRecord.cpp
    DEPENDS
        ${Boost_LIBRARIES}
        ${LOG4CXX_LIBRARIES}
        ${OpenCV_LIBS}
        Utils
        Messages
        GeoCoordinateSystem
        ImageProcessingCommonLib
     )
