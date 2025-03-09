/* ****************************************************************
 * Blob Target Detector
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Nov. 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
   *******************************************************************/

#ifndef VIDERE_DEV_TARGETDETECTOR_H
#define VIDERE_DEV_TARGETDETECTOR_H

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <RabitManager.h>
#include "VehicleInertialStatesMessage.h"
#include "CameraOrientationMessage.h"
#include "image_plus_metadata_message.h"
#include "ImageProcTargetInfoResultsMessage.h"
#include "config_data.h"
#include "logger.h"
#include "Utilities/ImagePixelLocationToRealWorldLocation.h"
#include "TargetParametersMessage.h"
#include "FeatureMatchProcCtrlMessage.h"
#include "Utilities/CameraOrientationValidation.h"

using namespace GeoCoordinateSystemNS;
using namespace VidereImageprocessing;
using namespace MathLibsNS;


namespace TargetDetectorNS
{

    //This is a top-level abstract class that all target detectors should
    //inhert from.  This helps to support multiple different target detection
    //processes including GPGPU accelerated processes.
    class TargetDetector
    {
    protected:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        ImageProcTargetInfoResultsMessage* _targetResultsMsg = nullptr;
        std::shared_ptr<ImagePixelLocationToRealWorldLocation> _pixeToRealWorldConvertor;

        //ToDo: this should be changed to a list of target parameters.
        std::shared_ptr<TargetParametersMessage> _targetType1ParamsMsg;
        std::shared_ptr<TargetParametersMessage> _targetType2ParamsMsg;
        std::shared_ptr<TargetParametersMessage> _targetType3ParamsMsg;
        std::shared_ptr<TargetParametersMessage> _targetType4ParamsMsg;

        //A range of Image Processing Parameters are in this message and
        //can be used to set target processing options.
        std::shared_ptr<FeatureMatchProcCtrlMessage> _imageProcessingParamsMsg;

        std::vector<cv::KeyPoint> _targetLocationInfo;

        CameraOrientationValidation _cameraOrientationValidation;

    public:

        TargetDetector(Rabit::RabitManager* mgrPtr,
                       std::shared_ptr<ConfigData> config);


        ImageProcTargetInfoResultsMessage* GetTargetResultsMsg()
        {
            return _targetResultsMsg;
        }

        void SetTargetResultsMsg(ImageProcTargetInfoResultsMessage* tgtResultMsg)
        {
            _targetResultsMsg = tgtResultMsg;
        }

        std::shared_ptr<ImagePixelLocationToRealWorldLocation> GetPixeToRealWorldConvertor()
        {
            return _pixeToRealWorldConvertor;
        }

        void SetPixeToRealWorldConvertor(std::shared_ptr<ImagePixelLocationToRealWorldLocation> iplrwo)
        {
            _pixeToRealWorldConvertor = iplrwo;
        }



        //Initialize the Target Detector here if required.
        virtual bool Initialize() {return true; }

        //If resources were setup in Initialized...
        //they can be relsease in the Close method.
        virtual void Close()  {};

        //Target Detectors may be holding resources such as cv:Mat memory
        //and other resourses that should be released when changing to another
        //image processing routine.
        virtual void ReleaseResources() {};

        //Check and update target and image processing prameters if necessary.
        virtual void CheckUpdateTargetParameters(bool forceUpdates = false) {};



        //Target Detection Method
        //Handles the Image Processing for detecting targets
        //and returns the results in the provide message.
        //Returns the number of targets found
        //A number less than zero indicates an error.
        virtual int DetectTargets(ImagePlusMetadataMessage* imagePlusMetaData,
                          ImageProcTargetInfoResultsMessage* targetResultsMsg,
                          std::shared_ptr<ImagePixelLocationToRealWorldLocation> pixeToRealWorldConvertor_sptr) = 0;


        //Check to see if there is an intermediate image (an image created as part of the
        //taget processing) available.  This image can be used for an output display.
        //Image numbers: [0...N)
        virtual bool IsIntermediateImageAvailable(int imgNumber) {return false;}

        //Get an intermediate image copy.  Return true if image is available and obtained.
        //return false if no image is available.
        //Image numbers: [0...N)
        virtual bool GetIntermediateImage(int imgNumber, cv::Mat &outImg) {return false;}

        //Mark the targets on the image.
        //If targetsOnly = true then only the final filtered targets
        //will be added... otherwise mark all targets found.
        virtual void MarkTargetsOnImage(cv::Mat &image,
                                        bool targetsOnly) {}

    };

}

#endif //VIDERE_DEV_TARGETDETECTOR_H
