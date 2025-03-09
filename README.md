[TOC]

NeuroGroove Videre
==================

This is the system for vision processing and control of the NeuroGroove racecar.
This software started from the USAFA UAS FalconVision/Videre code.
The software is designed to run on a NVidia Jetson TX2 or similar computer
and takes advantage of the NVidia CUDA GPGPU. 


Layout
======

* **mono_inteface** is the Mono/.NET code for interfaceing to the system a C# system.
* **protobuf_messages** contains the protobuf messages for the system
* **videre** is the c++ implementation of the vision processing system 
* **research** is the research and development work used to prove certain concepts before adding them to the vision system



