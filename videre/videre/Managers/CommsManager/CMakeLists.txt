include(rabit_add_manager)

rabit_add_manager(
    MANAGER CommsManager
    NAMESPACE videre
    MESSAGES_CONFIG "comms_mgr_config.yml"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES
        CommsManager.h
        command_response_message_handler.h
        CommUtils.h
        TelemetryMessageProcess.h
        zeromq_wrapper.h
        zeromq_wrapper_exception.h
    SRCS
        CommsManager.cpp
        command_response_message_handler.cpp
        CommUtils.cpp
        TelemetryMessageProcess.cpp
        zeromq_wrapper.cpp
    DEPENDS
        ${ZMQ_LIBRARIES}
        ${PROTOBUF_LIBRARIES}
        ${OpenCV_LIBS}
        Utils Messages
        ProtobufMessages
        GeoCoordinateSystem
        NemaTodeGpsParser
        ImageProcessTypes
        ImageProcessingCommonLib
    )
