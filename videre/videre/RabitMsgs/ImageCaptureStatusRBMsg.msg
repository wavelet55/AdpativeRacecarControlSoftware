# Image Capture Status
# Status of the Image Capture Manager.

# The Image Capture Enabled/Disabled Status:
# If there is a error or NumberOfImagesToCapture has
# been reached, or the images have been exhausted,
# this will be false.
bool ImageCaptureEnabled

# Set to true when the number of images capture
# equals the NumberOfImagesToCapture (assuming
# NumberOfImagesToCapture > 0);
bool ImageCaptureComplete

# Set to true if the source of images is exhausted.
# which can occur if images are being pulled from a
# directory of images.
bool EndOfImages

# Total Number of Images Captured Since Start of Videre
uint32 TotalNumberOfImagesCaptured

# Total Number of Images Captured Since Image Capture
# Enabled... Gets reset to zero when image capture is
# disabled.
uint32 CurrentNumberOfImagesCaptured

# Average Frames per second base on
# CurrentNumberOfImagesCaptured / Time since Last Enabled.
float64 AverageFramesPerSecond

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

# The Error Number will be non-zero if there is an
# error in the image capture process.  The error
# number may be used to indicate what the error is.
#   enum ImageCaptureError_e
#   {
#       ImageCaptureError_None = 0;
#       ImageCaptureError_SourceConfig = 1;
#       ImageCaptureError_SourceCapture = 02;
#   };
uint8  ErrorCode