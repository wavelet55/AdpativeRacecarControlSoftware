# Image Compression Control Message.
# Establishes Image compression parameters
# and controls whether or not image compression is used.

# Supported Image Compression Types.

# VisionImageCompressionType_e
#    jpeg = 0;
#    mpeg = 1;
#    gif = 2;

# Enable or Disable Image Compression
bool EnableImageCompression

# Enable or Disable Transmission of Compressed images.
# Note:  Image Compression could be enabled just to log
# the compressed images... not to transmit the images.
bool TransmitCompressedImages

# Set the Type of Image Compression, with VisionImageCompressionType_e above
uint8 ImageComressionType

# Set the approximate image compression ratio.
# The image compression ratio.
float64 ImageCompressionRatio

# The Desired framerate through the image compression
# process.  The actual framerate may be less if compression
# time or other factors limit the framerate.

# The image compression ratio.</value>
float64 FrameRate
