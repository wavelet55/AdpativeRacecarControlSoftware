#Enable or disable the Sip-n-Puff integral value
#output.  When disabled the integral value is cleared.
bool EnableSipnPuffIntegration

#Sip-n-Puff Gains... for the integrator
#When 1.0 it will take approximately 1 second
#to integrate from 0 to 100% with the the Sip or Puff
#output at 100%.
float64 SipnPuffBlowGain
float64 SipnPuffSuckGain

#When the SipnPuff value is less than the
#dead-band the output will be forced to zero.
float64 SipnPuffDeadBandPercent
