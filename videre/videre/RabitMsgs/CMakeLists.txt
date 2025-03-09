project(RabitMsgs)

include(rabit_add_messages)

# produces the rabit_msgs library

#=============================================================================
#              Add All Auto Generated Messages Here
#
#  -   NAMESPACE: is the namespace you want your messages to be in
#  -  H_TEMPLATE: is the *.h.em file used for generating messages headers
#  -CPP_TEMPLATE: is the *.cpp.em file used for generating messages sources
# -CSHP_TEMPLATE: if defined, system will output csharp messages
#  -  DEPRICATED: place all depricated messages here
#=============================================================================
rabit_add_messages(
    NAMESPACE "videre"
    H_TEMPLATE "msg.h.em"
    CPP_TEMPLATE "msg.cpp.em"
    CSHP_TEMPLATE "msg.cs.em"
    CS_NAMESPACE "VisionBridge.Messages"
    CSHP_OUTDIRECTORY ${CMAKE_SOURCE_DIR}/../mono_interface/VisionBridge/RabitMsgs/
    PYTHON_OUTDIRECTORY ${CMAKE_SOURCE_DIR}/../PythonRabitMessages
    PYTHON_TEMPLATE "msg.py.em"
    MESSAGES
        DTX_IMU_SensorSetCommandMsg.msg
        DTX_IMU_SensorSetValuesMsg.msg
        DTX_IMU_SystemCommandMsg.msg
        DTX_IMU_SystemStatusMsg.msg
        host_conn_rep.msg
        host_conn_req.msg
        MgrDtxConnectionStatusMsg.msg
        nexusBCIControlConfigMsg.msg 
        nexusBCIThrottleControlMsg.msg 
        sipnPuffBCIMsg.msg 
        sipnPuffConfigMsg.msg 
        VidereRecordControlMsg.msg
        VidereRecordStatusMsg.msg
		TstGlyphDetectionParametersRBMsg.msg
		#ImageCaptureControlRBMsg.msg
		#ImageCaptureStatusRBMsg.msg
		#ImageCompressionControlRBMsg.msg
		#StreamControlRBMsg.msg
    DEPRICATED
        none.msg
    )

#=============================================================================

#******************IMPORTANT TO KEEP THIS LINE IN EACH PROJECT****************
# This line is to let the rest of the build system know what messages 
# are being used
set(RABIT_SYSTEM_MESSAGES ${RABIT_SYSTEM_MESSAGES} PARENT_SCOPE)





