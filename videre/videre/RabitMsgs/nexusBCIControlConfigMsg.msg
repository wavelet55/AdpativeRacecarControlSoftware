#Enable the Nexus BCI Throttle Control
bool EnableNexusBCIThrottleControl

#Enable or disable the Sip-n-Puff for Throttle/Brake Control
bool EnableSipnPuffThrottleBrakeControl
bool SipnPuffBrakeOnlyControl

#The BCI Throttle Integrator Gain which controls how fast the
#throttle ramps up when the driver is thinking throttle On
float64 BCIThrottleIntegrationGain

#Delay time before the throttle/car speed ramps down after the
#driver thinks thottle off.  This is only used when the
#EnableSipnPuffThrottleBrakeControl and SipnPuffBrakeOnlyControl are false.
float64 BCIThrottleRampDownDelaySeconds

#The BCI Throttle Integrator Gain which controls how fast the
#throttle ramps down when the driver is thinking throttle Off
#This is only used when the
#EnableSipnPuffThrottleBrakeControl and SipnPuffBrakeOnlyControl are false.
float64 BCIThrottleRampDownIntegrationGain
