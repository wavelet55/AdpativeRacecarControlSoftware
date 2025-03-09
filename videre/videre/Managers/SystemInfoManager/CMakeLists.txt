include(rabit_add_manager)

rabit_add_manager(
    MANAGER SystemInfoManager
    NAMESPACE videre
    MESSAGES_CONFIG "sysinfo_mgr_config.yml"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES 
        system_info_manager.h
    SRCS

    DEPENDS
        Utils Messages
    )
