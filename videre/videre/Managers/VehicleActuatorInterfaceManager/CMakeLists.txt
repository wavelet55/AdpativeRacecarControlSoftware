include(rabit_add_manager)

rabit_add_manager(
    MANAGER VehicleActuatorInterfaceManager
    NAMESPACE videre
    MESSAGES_CONFIG "vehicleactuator_mgr_config.yml"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES
        VehicleActuatorInterfaceManager.h
        DceEpasSteeringInterface.h
        EpasSteeringDataRecords.h
        KarTechLinearActuator.h
        KarTechLADataRecords.h
    SRCS
        VehicleActuatorInterfaceManager.cpp
        DceEpasSteeringInterface.cpp
        EpasSteeringDataRecords.cpp
        KarTechLinearActuator.cpp
        KarTechLADataRecords.cpp
    DEPENDS
        ${Boost_LIBRARIES}
        ${LOG4CXX_LIBRARIES}
        ${OpenCV_LIBS}
        Utils
        Messages
        GeoCoordinateSystem
        ImageProcessingCommonLib
    )
