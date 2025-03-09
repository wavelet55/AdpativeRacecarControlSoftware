include(rabit_add_manager)

rabit_add_manager(
    MANAGER HeadOrientationManager
    NAMESPACE videre
    MESSAGES_CONFIG "headorientation_mgr_config.yml"
    WORKSPACE_H_TEMPLATE ${RBX_TOOLS}/templates/ws.h.em
    WORKSPACE_CPP_TEMPLATE ${RBX_TOOLS}/templates/ws.cpp.em
    INCLUDES
        HeadOrientationManager.h
        HeadOrientationDataRecord.h
    SRCS
        HeadOrientationManager.cpp
        HeadOrientationDataRecord.cpp
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
