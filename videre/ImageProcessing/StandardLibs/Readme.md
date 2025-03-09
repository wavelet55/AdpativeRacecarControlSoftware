[TOC]

Standard Image Processing Libraries
===================================

The Standard libraries do not require CUDA or other GPGPU specific
libraries or buid tools.

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
* [opencv](www.opencv.org/downloads.html)


Layout
======

