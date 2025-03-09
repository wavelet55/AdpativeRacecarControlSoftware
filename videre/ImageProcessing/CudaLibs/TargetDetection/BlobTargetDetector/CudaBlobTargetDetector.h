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

#ifndef CUDA_BLOBTARGETDETECTOR_H
#define CUDA_BLOBTARGETDETECTOR_H

#include "Blob.h"
#include "BlobResult.h"
#include <opencv2/core.hpp>
#include "BlobDetectorFixedParameters.h"
#include "CommonImageProcTypesDefs.h"
#include "../../CudaCommon/Utils/helper_cuda.h"

using namespace ImageProcLibsNS;
using namespace CommonBlobTargetDetectorNS;

namespace CudaBlobTargetDetectorNS
{

    class CudaBlobTargetDetector
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        double corner[4][2];        // pixLoc[2] and pixLoc[3] have garbage values initially

        // Blob initialization

        CBlobResult_cuda blobsBG, blobsFG, blobsH, blobsF;
        CBlob_cuda *currentBlob;

        //A temporary computed image.   This cv::Mat
        //is managed by the Blob Target Detector.
        cv::Mat *_imgTmp1Chan;

        ImageMaskRect_t _mask;

        ColorThresholds_t _colorThresholds;

        //Cuda Side Memory Resources

        //The Cuda side copy of the input image.
        //This memory buffer is kept from one image processing cycle
        //to the next to prevent constant allocting and de-allocating the image
        unsigned char *_cudaInpImgRGB = 0;
        size_t _cudaInpImgRGBMemSizeBytes = 0;

        //An interal Gray-Scale/1Chan Image
        //Used as a temp image buffer... the Image
        //buffer is kept from one image processing cycle to the
        //next to prevent constant allocting and de-allocating the image
        unsigned char *_cudaInternalTmpImg1Chan = 0;
        size_t _cudaInternalTmpImg1ChanMemSizeBytes = 0;

    public:
        // Zoomed in or not?
        bool IsZoomedIn = false;

        // Using background modeling data?
        bool BackgroundModelingEnabled = false;

        // Removing sunlight reflected areas?
        bool SunlightReflectionEnabled = false;

        // Ignoring wheel occlusion?
        bool WheelRemovalEnabled = false;

        // Mask out a portion of the image from processing?
        bool ImageMaskEnabled = false;

        // Mark found targets in red on image if set
        bool MarkTargetEnabled = false;

        bool DetectType1Targets = true;
        bool DetectType2Targets = true;
        bool DetectUnknownTargets = true;

        //Set to true for IR Sensors.
        //Otherwise an RGB EO sensor is assumed.
        bool IsIRSensor = false;

        // For 3 modes, nR mean, nG mean, and its standard deviation
        double BGMode[9];

        double SunOrientation[2];    // Azimuth and elevation

        //SensorType_e SensorType = SensorType_e::EO;

        int NumberOfType1Blobs = 0;
        int NumberOfType2Blobs = 0;
        int NumberOfType3Blobs = 0;

    public:
        CudaBlobTargetDetector();

        ~CudaBlobTargetDetector();

        //Initialize resources as need.
        //Returns false if ok, true if there was an error.
        bool Initialize();

        //Close out resources used by the Cuda Target Detectory
        void Close();


        //Target Detection Method
        //Handles the Image Processing for detecting targets
        //and returns the results in the provide message.
        //Returns the number of targets found
        //A number less than zero indicates an error.
        int DetectTargets(cv::Mat *imgInpRGB,
                          std::vector<BlobTargetInfo_t> *tgtResults);


        bool ImageBackgroundRemovalProcess(cv::Mat *imgInp, cv::Mat *imgOut);

    };


}
#endif //CUDA_BLOBTARGETDETECTOR_H
