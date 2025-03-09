# Image Capture Control
# Controls Image Capture and provides configuration
# parameters for Image Capture.
# Note:  The configureation parameters are only read
# and updated/set by Videre when the Image Capture is
# disabled.

# The Image Capture On flag must be true
# to capture images and do any of the processing
# of images.  Setting this flag to false will disable
# all other image processing.
# Note:  The configureation parameters are only read
# and updated/set by Videre when the Image Capture is
# disabled.

bool ImageCaptureEnabled

# If NumberOfImagesToCapture is greater than zero,
# the Vision System will capture the set number of images,
# process the images, and then disable ImageCapture.
# If NumberOfImagesToCapture is zero or less... the Vision
# System will continue capturing and processing images until
# ImageCaptureEnabled is disabled by the user.
# To Restart Image Capture after the NumberOfImagesToCapture
# has been reached... Disable and then re-enable Image
# capture with the ImageCaptureEnabled flag.

uint32 NumberOfImagesToCapture


# Desired Frames per second.
# In general the frame rate will be controlled by the time
# to do the image processing.  If image processing is disabled
# or very quick, this value can be used to slow down the image
# capture and processing rate.  Set to a higher value to get the
# max frame rate that is supported by the image processing time.

float64 DesiredFramesPerSecond

# Desired Image Width and Height
# Where Image Width and Height can be controlled, use
# these parameters.  If set to zero, the Videre Config
# info will be used.

uint32  DesiredImageWidth
uint32  DesiredImageHeight

# Source of Images for Image Capture
#   enum ImageCaptureSource_e
#   {
#       ImageCaptureSource_NoChange = 0;        //Don't change the current source.
#       ImageCaptureSource_OpenCVWebCam = 1;    //OpenCV Driver for WebCam
#       ImageCaptureSource_IPMFiles = 2;        //ImagePlusMetadata Files (.ipm)
#       ImageCaptureSource_CImageFiles  = 3;    //Compressed Image Files (.jpg)
#       ImageCaptureSource_Sensoray2253  = 4;
#       ImageCaptureSource_NVidiaCSI= 5;
#       ImageCaptureSource_VideoFile= 6;
#   }
uint8 ImageCaptureSource


# Image Capture Format:
#    MJPEG, YUV... Depends on webcam capability
#   enum CPImageCaptureFormat_e
#   {
#      Unknown = 0;
#      Grey8 = 1;
#      Grey16 = 2;
#      RGB24 = 3;
#      BGR24 = 4;
#      MJPEG = 5;
#      YUV422 = 6;
#   }
uint8 ImageCaptureFormat


# Primary Configuration String for the ImageCaptureSource.
# This could be the Device number for the WebCam,
#  or it could be the Directory of Image Files.
# If this is empty the Videre Config info will be used.
string ImageCaptureSourceConfigPri

# Secondary Configuration String for the ImageCaptureSource.
# This could be the Device number for the WebCam,
#  or it could be the Directory of Image Files.
# If this is empty the Videre Config info will be used.
string ImageCaptureSourceConfigSec

# When images are being captured from a finite
# source such as a directory of image files,
# if this flag is true, Image capture will restart
# capture from the start of the source after reaching
# the end.
bool ImageSourceLoopAround

bool AutoFocusEnable