include(rabit_system_configuration)

rabit_system_configuration(
    CONFIG_NAME "VIDERE DTX IMU CONFIGURATION"
    MANAGERS
        ImageCaptureManager
        VisionProcessManager
        StreamRecordManager
        CommsManager
        GPSManager
        DTX_IMU_InterfaceManager
        IMUCommManager
        HeadOrientationManager
        SipnPuffManager
        VehicleActuatorInterfaceManager
        VehicleStateManager
        VidereSystemControlManager
)