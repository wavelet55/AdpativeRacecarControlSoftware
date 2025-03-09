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


#ifndef VIDERE_DEV_FEATUREMATCHINGPROCESSOR_H
#define VIDERE_DEV_FEATUREMATCHINGPROCESSOR_H

#include "VisionProcessResources.h"
#include "VisionProcessorAbstract.h"
#include "TargetDetectorProcessControl.h"
#include "Utilities/JpgFileHandling.h"
#include "CameraCalibration/CameraCalibrationWith2DObjects.h"
#include "logger.h"
#include "CameraCalibrationData.h"
#include "StdCommon/ImageKeypointsAndFeatures.h"
#include "StdCommon/ImageFeatureMatcher.h"
#include <RabitStopWatch.h>

namespace videre
{
    //The Feature Matching Processor is largely used to help develop
    //and test image processing algorithms within the Videre / NVidia
    //enviroment.  This processor allows the user to capture images
    //that contain features of interest.  Then the features can be compared
    //to or matched with images from any input image stream.
    //A range of controls set in the FeatureMatchProcCtrlMessage will
    //allow the user to control the processing done.  It is expected
    //that this FeatureMatchingProcessor will expand over time to help
    //in the development and testing of a whole range of image processing
    //algorithms.
    class FeatureMatchingProcessor  : public VisionProcessorAbstract
    {
    private:


        FeatureMatchingState_e _FMPState;

        std::string _featureImageDirectoryName;
        std::string _featureImageBaseName;
        std::string _featureDataDirectoryName;
        std::string _featureDataFileName;
        ImageFormatType_e _imageFormatType;

        //The primary keypoints and Features is the set of keypoints
        //and features new images or query images will be compared against.
        ImageProcLibsNS::ImageKeypointsAndFeatures _imageKeyptsFeaturesPrimary;
        ImageProcLibsNS::ImageKeypointsAndFeatures _imageKeyptsFeaturesQuery;
        ImageProcLibsNS::ImageFeatureMatcher _imageFeatureMatcher;

        //The primage image is the image assocated with the _imageKeyptsFeaturesPrimary;
        cv::Mat _imagePrimary;

        int _numberOfFeatureImages = 0;
        std::vector<boost::filesystem::path> _listOfFearueImageFiles;

        bool _capturedImageOk = false;

        bool _capturedImageSentForView = false;

        int _numberSendCapturedImageTries = 0;

        ImagePlusMetadataMessage *_currImgMsgPtr = nullptr;

        VidereImageprocessing::JpgFileHandler _JpgFileHandler;

        //Stopwatches for general purpose use
        RabitStopWatch _stopwatch1;
        RabitStopWatch _stopwatch2;

    public:
        FeatureMatchingProcessor(VisionProcessResources* vpResources);

        ~FeatureMatchingProcessor();

        std::string CreateImageFilename(int fileIdx);


        void WriteFeatureDataToFile();

        //Reset the Vision Processor.
        //Reset must return all resources obtained during operation...
        //such a image messages, target result messages and the like.
        virtual void Reset();

        //This is the primary call for running the Vision Processor
        //each time through the Manager's loop.
        virtual VPProcessResult_e ExecuteUnitOfWork();

        bool ProcessImageFeatures(ImagePlusMetadataMessage *ipmdMsg);

        bool MatchImageFeatures(ImageKeypointsAndFeatures queryInpKptsFeatures,
                                ImageKeypointsAndFeatures matchSetInpKptsFeatures);

        bool PostProcessImageToBeSentOut(ImagePlusMetadataMessage *ipmdMsg);

    };

}
#endif //VIDERE_DEV_FEATUREMATCHINGPROCESSOR_H
