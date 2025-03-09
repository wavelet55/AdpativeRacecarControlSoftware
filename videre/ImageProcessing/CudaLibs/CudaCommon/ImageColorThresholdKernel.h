/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Dec. 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/


#ifndef VIDERE_DEV_IMAGECOLORTHRESHOLDKERNEL_H
#define VIDERE_DEV_IMAGECOLORTHRESHOLDKERNEL_H

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "Utils/helper_cuda.h"
#include "BlobDetectorFixedParameters.h"
#include "CommonImageProcTypesDefs.h"


namespace CudaImageProcLibsNS
{


    bool ImageColorThreshold(const cv::cuda::GpuMat &colorImageIn, cv::cuda::GpuMat &bwImageOut,
                             PixelColorValue_t &minColorValue, PixelColorValue_t &maxColorValue,
                             bool syncGPU, const cv::cuda::GpuMat *maskImage = nullptr);



    //The ImageColorThresholdKernel creates a black & white image
    //from a color (BGR) image.  The white regions are based on
    //all colors that fall within the given color range... the
    //black is all areas of the image that do not fall within the
    //color range.
    class ImageColorThresholdKernel
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

    public:

        ImageColorThresholdKernel() {}

        ~ImageColorThresholdKernel() {}



    };

}
#endif //VIDERE_DEV_IMAGECOLORTHRESHOLDKERNEL_H
