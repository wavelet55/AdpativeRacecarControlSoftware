#!/bin/bash
#Startup the CAN Bus on the Rudi TX2 Computer
#Sets the CAN bus for a 250K Baud
#This must be run before starting Videre.
#This script requires root access (sudo)
#If permission issue run with:  sudo -s 
#then 'exit' afterwords

ip link set can0 type can bitrate 250000
ip link set up can0

#Also setup the GPIO Pins on the Rudi TX2
#The pin mapping on the IO Connector is as follows:
#   Pin   GPIO#   TX2-internal Pin#
#    5    GPIO1   232       Output
#    6    GPIO2   233 	    Input
#    7    GPIO3   234       Output
#    8    GPIO4   235       Input

echo 232 > /sys/class/gpio/export
echo 233 > /sys/class/gpio/export
echo 234 > /sys/class/gpio/export
echo 235 > /sys/class/gpio/export

echo out > /sys/class/gpio/gpio232/direction
echo in > /sys/class/gpio/gpio233/direction
echo out > /sys/class/gpio/gpio234/direction
echo in > /sys/class/gpio/gpio235/direction

#set the inputs to be active low... read 1 when
#pin is pulled low.
echo 1 > /sys/class/gpio/gpio233/active_low
echo 1 > /sys/class/gpio/gpio235/active_low

#GPIO-1 is being used for a pull-up voltage...
#force the output high.
echo 1 > /sys/class/gpio/gpio232/value

