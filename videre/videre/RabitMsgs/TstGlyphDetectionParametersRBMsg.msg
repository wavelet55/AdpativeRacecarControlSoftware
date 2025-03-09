#Reference GlyphDetectionParameters_t
int32 Canny_low
int32 Canny_high
int32 GlyphAreaPixels_min
int32 GlyphAreaPixels_max
int32 NumberOfIterations
float64 ReprojectionErrorDistance
float64 ConfidencePercent           #[0, 100.0]

#Set the Glyph Processing Display Type
uint8  GlyphDetectorImageDisplayType

#Different Glyph Models can be used. This is an index to the desired
#model set.
uint8 GlyphModelIndex
