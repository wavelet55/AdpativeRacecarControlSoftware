[TOC]

Adaptive Racecar Control Software
=================================

DireenTech Inc. in conjunction with Falci Adaptive Motorsports adapted an 850 hp racecar donated by Furniture Row Racing (FRR), so that a person with quadriplegia (paralyzed from the neck down) can drive the car under head controls.  In further work with the University of Miami Miller School of Medicine the car control system was adapted to tie in a patient’s brain implant to augment the control of the racecar (BCI – Brain Computer Interface).  A couple of youtube videos show the system in operation:

   Smart Driving Cars Summit 2019, https://www.youtube.com/watch?v=8VZEUsgyoas
   Falci Adaptive Motorsports: Innovate 78: https://www.youtube.com/watch?v=hLZX7vdy3S4&t=64s
 
 Details of the adaptive racecar can be found in the paper: Harry, Randal, and James Direen, “Head-Controlled Racecar for Quadriplegics”, IEEE American Controls Conference, July (2020)

The primary adaptive racecar control software is provided here for study purposes. The code is provided as is without any warranties or guarantees. While we have had a great deal of success with the racecar, we strongly advise against attempting to use this software and design in another adaptived vehicle.  We made use of professional drivers as safety drivers whenever the racecar was driven under head control using this softare.   

This code was designed to run on an NVidia Jetson computer (TX2).  There is a separate microcontroller board that contains the accellerometers and gyroscopes used in the head control steering process.  This code communicates to the microprocessor board over a RS-233 / UART interface.  The firmware for the microprocessor board is not part of this repository.  The firmware may be obtained by contacting Harry Direen:  hdireen@direentech.com.

Videre (To See)
===============

This is the system for vision processing and the control of the NeuroGroove racecar.
 
VIDERE uses the DireenTech Rabit multi-threaded management system.
A copy of the latest Rabit system can be found at:
https://github.com/rdireen/rabitcpp
Rabit is a library for building multithreaded systems. 
Rabit works with managers, where each manager runs in its own thread.
Rabit has a publish-subscribe and queue-based messaging system for 
safely sharing information between the managers.

Library Dependancies
====================
When we can, we try to use the Ubuntu repos to get our dependencies. 
Right now it only looks like we have to do something special with 
*zeromq*, *cuda* and *opencv*

Try to use *checkinstall* when possible instead of *make install*. 
This will make it possible to easily uninstall packages with *dpkg*. 
Also, don't forget to run *ldconfig* after libraries have been installed 
so the machine knows where to find those libraries during the 
linking process.

List
----
* [zeromq](zeromq.org/area:download) download version: ZeroMQ 4.2.0
   ZeroMQ (0MQ) is the library used for communications between Videre and HOPS
   and Videre and the Videre Monitor program.  It is a fairly powerful, open source
   communications module.  Reference the documentation at: zeromq.org
   Ref: zeromq.org/intro:get-the-software for instuctions on downloading and building zmq.
   Note: starting with version 3.0 and above, the zmq.hpp header file was moved to a new
   git repository:  https://github.com/zeromq/cppzmq
   Use git clone https://github.com/zeromq/cppzmq  userdirectory
   to obtain the latest zmq.hpp file.  Copy this file to:  /usr/local/include
   (sudo cp zmq.hpp /usr/local/include)

* [protobuf] For older versions see:
        (https://developers.google.com/protocol-buffers/?hl=en) get by *sudo apt-get install libprotobuf-dev* 
* [protoc]  or older versions see
        (https://developers.google.com/protocol-buffers/?hl=en) get by *sudo apt-get install protobuf-compiler*

* [log4cxx](https://launchpad.net/ubuntu/+source/log4cxx) get by *sudo apt-get install liblog4cxx-dev*
* [boost](www.boost.org) get by *sudo apt-get install libboost-all-dev*-j4
* [cuda](https://developer.nvidia.com/cuda-downloads) (This is loaded with Jetpack don't reload)
* [opencv] Feb 2022, upgraded to opencv 4.5.   The instructions below basicallly work, just pull/checkout
        opencv version 4.5.5
        See:  OpenCV_4_CompileInfoArgs.txt
        (www.opencv.org/downloads.html) (This is loaded with Jetpack... Jetpack does not load a full version
        of opencv.  To build a full version of opencv (Jetson Xavier) use the instructions at:
        https://www.jetsonhacks.com/2018/11/08/build-opencv-3-4-on-nvidia-jetson-agx-xavier-developer-kit/

* [libsocketcan] (https://git.pengutronix.de/git/tools/libsocketcan)
             This library is used for the CAN bus interface.
             Documentation: https://www.kernel.org/doc/Documentation/networking/can.txt
             
             The can-utils package source code can be found at https://github.com/linux-can/can-utils. 
             This has great C code examples on how to read and write messages to the SocketCAN interface.
             
              build:
              git clone https://git.pengutronix.de/git/tools/libsocketcan
              ~/libsocketcan$ ./autogen.sh
              ~/libsocketcan$ ./configure
              ~/libsocketcan$ make
              ~/libsocketcan$ sudo make install
 
                            
* [Armadillo]
        http://arma.sourceforge.net/
        Armadillo is a high quality linear algebra library (matrix maths) for the C++ language, aiming towards a good balance between speed and ease of use.
        
        Conrad Sanderson and Ryan Curtin. 
        Armadillo: a template-based C++ library for linear algebra. 
        Journal of Open Source Software, Vol. 1, pp. 26, 2016. 
        
        Conrad Sanderson and Ryan Curtin. 
        A User-Friendly Hybrid Sparse Matrix Class in C++. 
        Lecture Notes in Computer Science (LNCS), Vol. 10931, pp. 422-430, 2018.
           
        The NVidia TX2 computers have cuBlas and NVBlas installed on them with the JetPack installation.

	    Follow the directions in the Armadillo Readme file for installation.
                        
            
* [gtest](https://code.google.com/p/googletest) get by *sudo apt-get install libgtest-dev* then run cmake in 
            /usr/src/gtest then copy *.a files to /usr/libscon

Note that CUDA has to be installed before OpenCV. In CMake, you must enable the options for 
CUDA if you want to use the CUDA versions of the OpenCV code.

Image Processing in Videre relies on OpenCV version 3.1 or greater. 
OpenCV 3  Reference:  http://docs.opencv.org/3.0-beta/modules/cuda/doc/introduction.html
In order to bring in some of the newere features, OpenCV has to be downloaded from its source and 
built for the Jetson board.  Directions for doing this can be found at:
Get the build script at:  https://github.com/jetsonhacks/buildOpenCVTX2
Follow the instructions there.


For camera calibration and other image processing functions, the following addtional libraries
are required:
* [libpopt]  *sudo apt-get install libpopt-dev*

* [rabitcpp] Part of the project files.  The libary is at:
        https://gitlab.direentech.com/Direentech_Internal/rabitcpp   (cmake and then install)

* [Fast DDS/RTPS] (https://www.eprosima.com/index.php/products-all/eprosima-fast-dds) 
  A script is contained in the Tools directory for installing fast-dds on a linux machine.

  The code generator requires Python 3.6 or higher with the following packages:
  pip3 install --user wheel
  pip3 install --user empy
  pip3 install --user pyros-genmsg
  pip3 install --user packaging
  pip3 install PyYAML

Build/Compile
=====

Cmake is use for the build/comile process.
To build/compile the project (from command line):
1) create a "build" directory under the videre directory
2) switch to the build directory
3) issue the command:  cmake -DOption1=ON/OFF -DOption2=ON/OFF ..
   Options:
    a)  WITHCUDA=ON OR OFF   Used to build on NVidia systems that support CUDA
    b)  WITHTK1=ON OR OFF    Used to set NVidia TK1 board compile options
    c)  WITHTX1=ON OR OFF    Used to set NVidia TX1 board compile options
    d)  WITHTX2=ON OR OFF    Used to set NVidia TX2 board compile options
    d)  WITHXAVIER=ON OR OFF Used to set NVidia Xavier board compile options
    e)  WITHTESTS=ON OR OFF  Used to compile unit tests
    f)  WITHCOVERAGE=ON or OFF Compile code with code coverage (for online CI)

4) issue the command:  make

Autorun Service
=====
Put the files:
    nascar_can_bus 
    nascar_vider
    
into the directory:  /etc/init.d 
Ensure the APPDIR in the files points to the correct file locations and
APPBIN is the appropriage binary filename
Change the files to executables with:  chmod +x

For the new daemons:
    1) sudo update-rc.d nascar_can_bus defaults
    2) sudo update-rc.d nascar_videre defaults
    
For control of the service:
    service nascar_videre (start|stop|restart|status)   
    
Put the file:
	99-usb-serial.rules

In the directory:  /etc/udev/rules.d
The file might need to be modified to match mapping of USB ports to USB serial devices.
From a terminal run the command:  sudo udevadm trigger
This maps USB ports to specific serial devices and gives them specific file names
that are used in the VidereConfig.ini
The Linux command:  udevadm info -a -n /dev/ttyUSB  
can be used to find the Kernal info used in the 99-usb-serial.rules file.

Add User to the Serial Comm Port(s) Dialout group:
        sudo usermod -a -G dialout $USER


Layout
======

Contributing
============
Reporting bugs, suggesting features, helping with documentation, and adding to the code is very welcome. 

License
=======

Copyright (C) 2025  Harry Direen
The Adaptive Racecar Control Software is licensed under GNU General Public License, version 3, a copy of this license has been provided within the COPYING file in this directory, and can also be found at <http://www.gnu.org/licenses/>.
 
