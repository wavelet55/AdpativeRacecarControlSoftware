include(rabit_system_configuration)

rabit_system_configuration(
    CONFIG_NAME "VIDERE CONFIGURATION"
    MANAGERS
        ImageCaptureManager
        VisionProcessManager
        StreamRecordManager
        CommsManager
        GPSManager
        IMUCommManager
        HeadOrientationManager
        SipnPuffManager
        VehicleActuatorInterfaceManager
        VehicleStateManager
        VidereSystemControlManager
)