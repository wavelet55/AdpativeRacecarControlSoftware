include(rabit_add_manager)

rabit_add_manager(
    MANAGER RemoteControlManager
    NAMESPACE videre
    MESSAGES_CONFIG "remotecontrol_mgr_config.ymll"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES
        RemoteControlManager.h
        RemoteControlRxMessageParser.h
    SRCS
        RemoteControlManager.cpp
        RemoteControlRxMessageParser.cpp
    DEPENDS
        ${Boost_LIBRARIES}
        ${LOG4CXX_LIBRARIES}
        ${OpenCV_LIBS}
        Utils
        Messages
    )
