# StreamRecordControlRBMsg

# Stream Images On/Off
bool StreamImagesEnabled

# This is the max frame rate for streaming images...
# The Vision System may be able to capture and process images
# at 30 fps or some rate higher than bandwidth will allow to
# stream images to the ground.  Setting this number to a lower number
# say 5 fps will reduce the image rate sent to the ground.
# If zero or less... stream images at max rate.
float64 StreamImageFrameRate



# ImageCompressionQuality
# Highest quality is 100, lowest is 1
# High Quality means low compression, lower quality
# means higher compression.
uint32 ImageCompressionQuality


# StreamImageScaleDownFactor
# The factor to scale the image down by before
# compressing the image and sending it out.
# A value of 2.0 will cut the image in half.
# The value must be >= 1.0.  1.0 will not change the
# image size.
float64 StreamImageScaleDownFactor
