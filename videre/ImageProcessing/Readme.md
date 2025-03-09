[TOC]

Image Processing Libraries
==========================

Image processing libraries are constructed speparate from the Videre 
system software.  By constructing the libraries separate from Videre and
the dependencies of Videre it will make the libraries easier to use
in other settins... such as being use in Python or other scripting 
languages. One of the primary reason for splitting out the image processing
libraries is to allow the GPGPU accelerated libraries to be built
using tool chains such as NVidia's Cuda development tools without
requiring all of Videre having to be built with the specialized tool
chains.

The Standard libraries do not require CUDA or other GPGPU specific
libraries or buid tools.

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

CUDA is only required for the CUDA accelerated image processing librries,
it is not required for the standard libraries.

Note that CUDA has to be installed before OpenCV. In CMake, you must enable the options for CUDA if you want to use the CUDA versions of the OpenCV code.


Layout
======

