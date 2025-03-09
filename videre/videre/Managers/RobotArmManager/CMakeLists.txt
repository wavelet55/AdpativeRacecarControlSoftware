include(rabit_add_manager)

rabit_add_manager(
    MANAGER RobotArmManager
    NAMESPACE videre
    MESSAGES_CONFIG "robotarm_mgr_config.yml"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES
        RobotArmManager.h
        RobotArm_RxMessageParser.h
        RobotArm_TxMessageFormatter.h
    SRCS
        RobotArmManager.cpp
        RobotArm_RxMessageParser.cpp
        RobotArm_TxMessageFormatter.cpp
    DEPENDS
        ${Boost_LIBRARIES}
        ${LOG4CXX_LIBRARIES}
        ${OpenCV_LIBS}
        Utils
        Messages
    )
