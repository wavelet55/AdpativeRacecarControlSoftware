[TOC]

Adaptive Racecar Control Software
=================================

DireenTech Inc. in conjunction with Falci Adaptive Motorsports adapted an 850 hp racecar donated by Furniture Row Racing (FRR), so that a person with quadriplegia (paralyzed from the neck down) can drive the car under head controls.  In further work with the University of Miami Miller School of Medicine the car control system was adapted to tie in a patient’s brain implant to augment the control of the racecar (BCI – Brain Computer Interface).  A couple of youtube videos show the system in operation:

   Smart Driving Cars Summit 2019, https://www.youtube.com/watch?v=8VZEUsgyoas
   Falci Adaptive Motorsports: Innovate 78: https://www.youtube.com/watch?v=hLZX7vdy3S4&t=64s
 
 Details of the adaptive racecar can be found in the paper: Harry, Randal, and James Direen, “Head-Controlled Racecar for Quadriplegics”, IEEE American Controls Conference, July (2020)

The primary adaptive racecar control software is provided here for study purposes. The code is provided as is without any warranties or guarantees. While we have had a great deal of success with the racecar, we strongly advise against attempting to use this software and design in another adaptived vehicle.  We made use of professional drivers as safety drivers whenever the racecar was driven under head control using this softare.   

This code was designed to run on an NVidia Jetson computer (TX2).  There is a separate microcontroller board that contains the accellerometers and gyroscopes used in the head control steering process.  This code communicates to the microprocessor board over a RS-233 / UART interface.  The firmware for the microprocessor board is not part of this repository.  The firmware may be obtained by contacting Harry Direen:  hdireen@direentech.com.


Layout
======

* **mono_inteface** is the Mono/.NET code for interfaceing to the system a C# system.
* **protobuf_messages** contains the protobuf messages for the system
* **videre** is the c++ the primare adaptive racecar control software including image processing 
* **CarDisplay** is a python based display interface used in the racecar


Contributing
============
Reporting bugs, suggesting features, helping with documentation, and adding to the code is very welcome. 

License
=======

Copyright (C) 2025  DireenTech Inc. (www.direentech.com)
The Adaptive Racecar Control Software is licensed under GNU General Public License, version 3, a copy of this license has been provided within the COPYING file in this directory, and can also be found at <http://www.gnu.org/licenses/>.
 

