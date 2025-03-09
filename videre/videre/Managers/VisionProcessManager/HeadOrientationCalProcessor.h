/* ****************************************************************
 * Head Orientation Calibration Processor
 * DireenTech Inc.  (www.direentech.com)
 * Athrs: Randy Direen PhD, Harry Direen PhD
 * Date: Aug. 2018
 *
 *
 * Notes:  Vedere measures the head position based upon a camera
 * in the car pointing at the head.  The camera measurement of the
 * head orientation has to be aligned with the car framework.
 * This cal. processor obtains images and head orientation in standard
 * positions and then determines the associated rotation quaternions
 * to bring the measurement in-line with the car.
  *******************************************************************/


#ifndef VIDERE_DEV_HEADORIENTATIONCALPROCESSOR_H
#define VIDERE_DEV_HEADORIENTATIONCALPROCESSOR_H

#include "VisionProcessResources.h"
#include "VisionProcessorAbstract.h"
#include "TargetDetectorProcessControl.h"
#include "HeadTrackingControlMessage.h"
#include "HeadTrackingOrientationMessage.h"
#include "../../Utils/logger.h"
#include "CameraCalibrationData.h"
#include "Quaternion.h"
#include "GlyphDetector/TrackHeadProcess.h"
#include "AccelerometerGyroMessage.h"
#include "HeadOrientationCalDataMessage.h"
#include "XYZCoord_t.h"
#include "GlyphDetector/GlyphModel.h"

namespace videre
{

#define NUMBEROFHEADCALORIENTATIONS 2

    class HeadOrientationCalProcessor : public VisionProcessorAbstract
    {

        CameraCalibrationState_e _CalState;

        std::string _calDataDirectoryName;
        std::string _calDataFileName;

        int _numberOfCalImages = 0;

        Quaternion_t CapturedHeadOrientations[NUMBEROFHEADCALORIENTATIONS];

        CudaImageProcLibsTrackHeadNS::GlyphModel _headModelData;

        CudaImageProcLibsTrackHeadNS::TrackHeadProcess _trackHeadProcess;

        std::shared_ptr<HeadTrackingControlMessage> _headTrackingControlMsg;

        std::shared_ptr<TrackHeadOrientationMessage> _trackHeadOrientationMsg;


        std::shared_ptr<AccelerometerGyroMessage> _accelGyroHeadMsg;

        std::shared_ptr<AccelerometerGyroMessage> _accelGyroFixedMsg;

        std::shared_ptr<HeadOrientationCalDataMessage> _headOrientationCalDataMsg;


        HeadOrientationCalData_t _headOrientationCalData;

        std::string _htParamsFilename = "HeadTrackingConfig.ini";

        std::string _headOrientationCalFilename = "HeadOrientationCal.ini";


        bool _capturedImageOk = false;

        bool SendTrackHeadDataOut = false;

        bool _capturedImageSentForView = false;

        int _numberSendCapturedImageTries = 0;

        ImagePlusMetadataMessage *_currImgMsgPtr = nullptr;

        ImageProcLibsNS::CameraCalibrationData  _cameraCalData;

        XYZCoord_t _capturedHeadAccelOrientation;
        XYZCoord_t _capturedFixedAccelOrientation;

        std::string _glyphModelDirectoryName;
        int _NoOfGlyphModels = 2;
        int _selectedGlyphModel = 0;

        std::string _glyphModelFileNames[MAXNUMBERGLYPHMODELS];
        int _glyphScaleVals[MAXNUMBERGLYPHMODELS];


    public:
        HeadOrientationCalProcessor(VisionProcessResources* vpResources);

        ~HeadOrientationCalProcessor();

        std::string getGlyphModelFilename(int modelIdx);

        //Intialize any resources at the start of operation.
        virtual void Intialize() final;


        //Reset the Vision Processor.
        //Reset must return all resources obtained during operation...
        //such a image messages, target result messages and the like.
        virtual void Reset();

        void WriteCalDataToFile();

        //This is the primary call for running the Vision Processor
        //each time through the Manager's loop.
        virtual VPProcessResult_e ExecuteUnitOfWork();

        bool PostProcessImageToBeSentOut(ImagePlusMetadataMessage *inputMsg,
                                           ImagePlusMetadataMessage *outputMsg);

    };

}
#endif //VIDERE_DEV_HEADORIENTATIONCALPROCESSOR_H
