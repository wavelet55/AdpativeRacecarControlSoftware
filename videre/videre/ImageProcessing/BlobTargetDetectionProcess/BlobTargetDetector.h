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

#ifndef VIDERE_DEV_BLOBTARGETDETECTOR_H
#define VIDERE_DEV_BLOBTARGETDETECTOR_H

#include <RabitManager.h>
//#include "VehicleInertialStatesMessage.h"
//#include "CameraOrientationMessage.h"
//#include "image_plus_metadata_message.h"
#include "ImageProcTargetInfoResultsMessage.h"
#include "config_data.h"
#include "logger.h"
#include "global_defines.h"
#include <opencv2/core.hpp>
#include "TargetDetection/BlobTargetDetectorOpenCVSimple/StdBlobTargetDetectorOpenCVSimple.h"
#include "Utilities/ImagePixelLocationToRealWorldLocation.h"
#include "TargetParametersMessage.h"
#include "TargetDetector.h"
#include "CommonImageProcTypesDefs.h"
//#include "TargetDetection/BlobTargetDetector/StdBlobTargetDetector.h"
#include "TargetDetection/BlobTargetDetectorOpenCVSimple/StdBlobTargetDetectorOpenCVSimple.h"
#include "BlobDetectorFixedParameters.h"
#include "FacialFeatureDetection/FacePoseDetector.h"

#ifdef CUDA
//#include "TargetDetection/BlobTargetDetector/CudaBlobTargetDetector.h"
#include "TargetDetection/BlobDetectorOpenCVSimple/CudaBlobTargetDetectorOpenCVSimple.h"
#include "PixelColorValue.h"
#endif

using namespace GeoCoordinateSystemNS;
using namespace VidereImageprocessing;
using namespace ImageProcLibsNS;
using namespace MathLibsNS;

namespace TargetDetectorNS
{

    //This class is a shell for the various Blob Target Detectors
    //located in ImageProcessing Libraries.  A number of different
    //blob detectors are available for test purposes.  The orginal
    //JHart/Dominator routine is available plus one or more versions
    //base upon the openCV blob detector.  NVidia GPU optimzed versions
    //are available if running on an Nvidia processor board.
    //This class instantiates one Blob Detector.
    //The class can be instantiated using a Standard or GPGUP/CUDA
    //accelerated blob target detector.  This class handles the interface
    //to the library routines and returns the results in the message structure
    //expected by Videre.
    //
    //There were two Blob Target Detectors to choose from.  The Old/Std one
    //is being dropped because is uses old opencv routines that are no longer
    //supported.  Right now I am cheating and selecting the newer OpenCVSimple 
    //blob detector when the old one is requested.
    class BlobTargetDetector : public TargetDetector
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _videreCfg;

        //Blob Target Detectors... only one will be used per BlobTargetDetector
        //Instantiation... the user can choose the GPGPU Accelerated version (Default)
        //or the standard non-accelerated version.

        bool _cudaTargetDetectorEnabled = false;

        //StdBlobTargetDetectorNS::StdBlobTargetDetector *_stdBlobTargetDetectorPtr = nullptr;

        StdBlobTargetDetectorOpenCVNS::StdBlobTargetDetectorOpenCVSimple *_stdBlobTargetDetectorOpenCVPtr = nullptr;

#ifdef CUDA
        //CudaBlobTargetDetectorNS::CudaBlobTargetDetector *_cudaBlobTargetDetectorPtr = nullptr;
        CudaBlobTargetDetectorOpenCVNS::CudaBlobTargetDetectorOpenCV *_cudaBlobTargetDetectorOpenCVPtr = nullptr;
#endif

    public:
        //A range of parameters used by the Blob Detector
        BlobDetectorParameters_t BlobDetectorParameters;

        //Results are placed in this structure.
        //The tgtResults may contain more targets found than
        //the final filtered number of targets.
        std::vector<BlobTargetInfo_t> TargetResults;

        PixelColorValue_t TargetColorVal;

    public:
        BlobTargetDetector(Rabit::RabitManager* mgrPtr,
                           std::shared_ptr<ConfigData> config,
                           bool useOpenCVBlobDetector,
                            bool useGPGPUAcceleration);

        ~BlobTargetDetector();

        //This flag will be true if useGPGPUAcceleration = true and
        //GPGPU is available, otherwise it will be false.
        bool IsGPGPUAccelerationEnabled()
        {
            return _cudaTargetDetectorEnabled;
        }

        void ReadBlobDetectorConfigParameters();

        virtual bool Initialize() override;

        virtual void Close();


        //Target Detectors may be holding resources such as cv:Mat memory
        //and other resourses that should be released when changing to another
        //image processing routine.
        virtual void ReleaseResources() override;

        //Check and update target and image processing prameters if necessary.
        virtual void CheckUpdateTargetParameters(bool forceUpdates = false) override;


        //Target Detection Method
        //Handles the Image Processing for detecting targets
        //and returns the results in the provide message.
        //Returns the number of targets found
        //A number less than zero indicates an error.
        virtual int DetectTargets(ImagePlusMetadataMessage* imagePlusMetaData,
                          ImageProcTargetInfoResultsMessage* targetResultsMsg,
                          std::shared_ptr<ImagePixelLocationToRealWorldLocation> pixeToRealWorldConvertor_sptr) override;


        bool ImageBackgroundRemovalProcess(cv::Mat *imgInp, cv::Mat *imgOut);


        int AddTargetsFoundToTargetList(int targetType, int maxNoTgtsToAdd = 0);

        //Check to see if there is an intermediate image (an image created as part of the
        //taget processing) available.  This image can be used for an output display.
        //Image numbers: [0...N)
        virtual bool IsIntermediateImageAvailable(int imgNumber) override;

        //Get an intermediate image.  Return true if image is available and obtained.
        //return false if no image is available.  An image copy is made if makecopy = true.
        //Image numbers: [0...N)
        virtual bool GetIntermediateImage(int imgNumber, cv::Mat &outImg) override;

        //Mark the targets on the image.
        //If targetsOnly = true then only the final filtered targets
        //will be added... otherwise mark all targets found.
        virtual void MarkTargetsOnImage(cv::Mat &image,
                                         bool targetsOnly) override;

    };

}
#endif //VIDERE_DEV_BLOBTARGETDETECTOR_H
