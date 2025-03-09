include(rabit_add_manager)

rabit_add_manager(
    MANAGER DTX_IMU_InterfaceManager
    NAMESPACE videre
    MESSAGES_CONFIG "dtx_imu_config.yml"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES 
        DTX_IMU_InterfaceManager.h
        DTX_IMU_RxMessageParser.h
    SRCS
        DTX_IMU_InterfaceManager.cpp
        DTX_IMU_RxMessageParser.cpp
    DEPENDS
        Utils Messages fastrtps fastcdr
    )
