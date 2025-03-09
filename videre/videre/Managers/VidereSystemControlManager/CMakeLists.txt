include(rabit_add_manager)

rabit_add_manager(
    MANAGER VidereSystemControlManager
    NAMESPACE videre
    MESSAGES_CONFIG "videre_sys_control_mgr_config.yml"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES
        VidereSystemControlManager.h
        HeadOrientationVehicleControlProcess.h
        SystemControlResources.h
        VidereSystemControlDataRecord.h
    SRCS
        VidereSystemControlManager.cpp
        HeadOrientationVehicleControlProcess.cpp
        SystemControlResources.cpp
        VidereSystemControlDataRecord.cpp
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
