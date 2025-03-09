/* ****************************************************************
 * Blob Target Detector
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Sept 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 * The Blob Target Detector was orginally developed for the JHART
 * and earlier Huntsman programs by Dr. Hyukseong Kwon in the June 2012
 * timeframe.   It was updated for the Dominator project by Dr. Gruber
 * in March 2014.
 *
 * The Blob Target Detector is being optimized to use NVidia's GPGPU
 * processor technology.  OpenCV has optimizations for NVidia's GPGPU
 * in addition other blob detection algorithms are being optimized
 * to use the GPGPU.
 *
 * The Blob Target Detector makes use of a Blob Library.  I do not have
 * the details of where this library came from other than the header
 * information in the files.  Modifications and optimizations to this
 * library are being made as necessary to suport the Blob Target Detector.
 *
  *******************************************************************/

#ifndef CUDA_BLOBTARGETDETECTOR_OPENCVSIMPLE_H
#define CUDA_BLOBTARGETDETECTOR_OPENCVSIMPLE_H

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "BlobDetectorFixedParameters.h"
#include "CommonImageProcTypesDefs.h"
#include "../../CudaCommon/Utils/helper_cuda.h"

using namespace ImageProcLibsNS;
using namespace CommonBlobTargetDetectorNS;

namespace CudaBlobTargetDetectorOpenCVNS
{

    class CudaBlobTargetDetectorOpenCV
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        //OpenCV Simple Blob Detector:
        cv::SimpleBlobDetector::Params _blobDetectionParams;

        cv::Ptr<cv::SimpleBlobDetector> _blobDetector;

        std::vector<cv::KeyPoint> _blobsLoc;


        double corner[4][2];        // pixLoc[2] and pixLoc[3] have garbage values initially


        //Cuda Side Memory Resources

        //The Cuda side copy of the input image.
        //This memory buffer is kept from one image processing cycle
        //to the next to prevent constant allocting and de-allocating the image

        //A temporary computed image.   This cv::Mat
        //is managed by the Blob Target Detector.
        cv::cuda::GpuMat _cudaImgBWBlobMat;

        //A Gaussian or otherwise Filtered Image.
        cv::cuda::GpuMat _cudaImgFiltedMat;

        cv::Ptr<cv::cuda::Filter> _gaussianFilter;

    public:
        //A range of parameters used by the Blob Detector
        BlobDetectorParameters_t BlobDetectorParameters;

        int NumberOfType1Blobs = 0;
        int NumberOfType2Blobs = 0;
        int NumberOfType3Blobs = 0;

    public:
        CudaBlobTargetDetectorOpenCV();

        ~CudaBlobTargetDetectorOpenCV();

        //Initialize resources as need.
        //Returns false if ok, true if there was an error.
        bool Initialize();

        //Close out resources used by the Cuda Target Detectory
        void Close();

        void releaseResources();

        //Target Detection Method
        //Handles the Image Processing for detecting targets
        //and returns the results in the provide message.
        //Returns the number of targets found
        //A number less than zero indicates an error.
        int DetectTargets(cv::cuda::GpuMat *imgInpBGR,
                          std::vector<BlobTargetInfo_t> *tgtResults);


        bool ImageBackgroundRemovalProcess(cv::Mat *imgInp, cv::Mat *imgOut);

        bool GetIntermediateImage(int imgNumber, cv::OutputArray outImg);

    };


}
#endif //CUDA_BLOBTARGETDETECTOR_OPENCVSIMPLE_H
