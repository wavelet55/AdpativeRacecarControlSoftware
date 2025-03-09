[TOC]

Image Processing Libraries
==========================

The CUDA libraries are designed to take adavante of the NVidia GPGPU
and require the CUDA development tool chain.

Library Dependancies
====================
The Image processing libraries should not depend on Videre or the Rabit
management system so that they can be used elsewhere.

Try to use *checkinstall* when possible instead of *make install*. 
This will make it possible to easily uninstall packages with *dpkg*. 
Also, don't forget to run *ldconfig* after libraries have been installed 
so the machine knows where to find those libraries during the 
linking process.

List
----
* [log4cxx](https://launchpad.net/ubuntu/+source/log4cxx) get by *sudo apt-get install liblog4cxx10-dev*
* [boost](www.boost.org) get by *sudo apt-get install libboost-all-dev*
* [cuda](https://developer.nvidia.com/cuda-downloads)
* [opencv](www.opencv.org/downloads.html)

CUDA is required for the CUDA accelerated image processing librries.

Note that CUDA has to be installed before OpenCV. In CMake, you must enable the options for CUDA if you want to use the CUDA versions of the OpenCV code.


Layout
======

