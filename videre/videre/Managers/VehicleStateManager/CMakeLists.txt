include(rabit_add_manager)

rabit_add_manager(
    MANAGER VehicleStateManager
    NAMESPACE videre
    MESSAGES_CONFIG "vehiclestate_mgr_config.yml"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES
        VehicleStateManager.h
        VehicleStateDataRecord.h
    SRCS
        VehicleStateManager.cpp
        VehicleStateDataRecord.cpp
    DEPENDS
        ${Boost_LIBRARIES}
        ${LOG4CXX_LIBRARIES}
        ${OpenCV_LIBS}
        Utils
        Messages
        GeoCoordinateSystem
        ImageProcessTypes
        ImageProcessingCommonLib
    )
