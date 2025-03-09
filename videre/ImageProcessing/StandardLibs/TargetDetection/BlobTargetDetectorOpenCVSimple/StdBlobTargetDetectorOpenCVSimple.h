/* ****************************************************************
 * OpenCV Lib Blob Target Detector
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Sept 2016
 * Updated Jan 2018
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *
  *******************************************************************/

#ifndef STANDARD_BLOBTARGETDETECTOR_OPENCV_H
#define STANDARD_BLOBTARGETDETECTOR_OPENCV_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "BlobDetectorFixedParameters.h"
#include "CommonImageProcTypesDefs.h"
#include "PixelColorValue.h"

using namespace ImageProcLibsNS;
using namespace CommonBlobTargetDetectorNS;

namespace StdBlobTargetDetectorOpenCVNS
{

    class StdBlobTargetDetectorOpenCVSimple
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        //OpenCV Simple Blob Detector:
        cv::SimpleBlobDetector::Params _blobDetectionParams;

        cv::Ptr<cv::SimpleBlobDetector> _blobDetector;

        std::vector<cv::KeyPoint> _blobsLoc;


        //A temporary computed image.   This cv::Mat
        //is managed by the Blob Target Detector.
        cv::Mat *imgBWBlobPtr = nullptr;

        //A Gaussian or otherwise Filtered Image.
        cv::Mat *imgFiltedPtr = nullptr;



    public:
        //A range of parameters used by the Blob Detector
        BlobDetectorParameters_t BlobDetectorParameters;

        int NumberOfType1Blobs = 0;
        int NumberOfType2Blobs = 0;
        int NumberOfType3Blobs = 0;

    public:
        StdBlobTargetDetectorOpenCVSimple();

        ~StdBlobTargetDetectorOpenCVSimple();

        //Create a blob-detector and load with basic parameters
        bool Initialize();

        //Release all resources.
        void Close();

        void releaseResources();

        //Target Detection Method
        //Handles the Image Processing for detecting targets
        //and returns the results in the provide message.
        //Returns the number of targets found
        //A number less than zero indicates an error.
        int DetectTargets(cv::Mat *imgInpBGR,
                          std::vector<BlobTargetInfo_t> *tgtResults);


        bool ImageBackgroundRemovalProcess(cv::Mat *imgInp, cv::Mat *imgOut);

        //Generate a output image based on pixels that are within a color range.
        //The input image can be from an EO or IR sensor.
        //The output image matrix is assumed to be of the shape and size to
        //hold the output image results.
        //A part of the image can be masked by providing a mask rectangle,
        //that part of the image will be black.
        bool ImageColorThreshold(const cv::Mat *imgInp,
                                 cv::Mat *imgOut,
                                 ImageMaskRect_t *maskImgRect = nullptr);

        bool GetIntermediateImage(int imgNumber, cv::OutputArray outImg);


    };

}
#endif //STANDARD_BLOBTARGETDETECTOR_OPENCV_H
