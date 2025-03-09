Sensoray 2253S
============

Video Driver  (Added April 11, 2017)
Harry Direen

Sensoray Version 1.2.14
Please see Linux software driver 2253 manual

Relies on Linux Driver:  sdk_x53_linux_1.2.14
This driver must be downloaded and installed on in order to use the Sensoray 2253S video input.
http://www.sensoray.com/products/2253.htm


From Manual:
The SDK has been developed on Linux Ubuntu LTS and support is provided
for this distribution. The SDK may work on other Linux distributions, but
this is not guaranteed. The lowest required kernel version is 2.6.25. An
effort has been made to support version 2.6.18, although some features
may be limited.

Basic Operation
When the driver is loaded, there are three video device nodes and one GPIO
char device created in the /dev directory. The video nodes are named
“videoX” where X depends on the number of video devices present in the
system. The first two video device nodes are for video capture/preview and
the third video device node is for video output.

Video Capture and Output Driver
The driver supports Video4Linux 2 (V4L2) ioctls. The V4L2 API is well
documented at the LinuxTV web site
(http://www.linuxtv.org/downloads/v4l-dvb-apis/). V4L2 operation is not
supported for kernels below 2.6.25. The v4l-dvb hg or git tree is not required
(and not recommended) to use this driver.

The three video devices can be used with applications that support V4L2 API.
Video can be captured using uncompressed YUV422 (packed YUYV or UYVY)
or YUV420SP (NV12 semi-planar, Y plane and interleaved CrCb plane) or
encoded in compressed formats JPEG, MPEG4 ASP or H.264 elementary
streams. The MPEG4 or H.264 streams can also be muxed with AAC audio
(with A/V sync) in a MP4 container or a MPEG transport stream. Both capture
devices record video from a single source, and each capture device can be
started, stopped, and configured independently. Some video options cannot
be configured independently, such as interpolation, brightness, hue, contrast,
and saturation 

Model 2253 device is accessed via /dev/videoX where X is the number
assigned to the device when it was plugged in. Three /dev/videoX nodes will
be created for each 2253 device. The first two video device nodes are for
video capture and the third is a video output. Which device the node
references can be determined by using the VIDIOC_QUERYCAP ioctl to
examine the card field for the serial no. The devices support multiple open,
allowing some parameters to be changed on-the-fly while another program is
accessing the device.

