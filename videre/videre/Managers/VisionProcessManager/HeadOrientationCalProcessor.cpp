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

#include "HeadOrientationCalProcessor.h"
#include "FileUtils.h"
#include <opencv2/core/core.hpp>
#include "CameraCalibration/CameraCalReaderWriter.h"
#include "Utilities/HeadTrackingCalParamsReaderWriter.h"

namespace videre
{

    HeadOrientationCalProcessor::HeadOrientationCalProcessor(VisionProcessResources* vpResources)
        : VisionProcessorAbstract(vpResources), _trackHeadProcess()
    {
        _trackHeadOrientationMsg = make_shared<TrackHeadOrientationMessage>();
        VPRptr->GetMgrPtr()->AddPublishSubscribeMessage("TrackHeadOrientationMessage", _trackHeadOrientationMsg);

        _headTrackingControlMsg = make_shared<HeadTrackingControlMessage>();
        VPRptr->GetMgrPtr()->AddPublishSubscribeMessage("HeadTrackingControlMessage", _headTrackingControlMsg);

        _accelGyroHeadMsg = std::make_shared<AccelerometerGyroMessage>();
        VPRptr->GetMgrPtr()->AddPublishSubscribeMessage("AccelerometerGyroHeadMsg", _accelGyroHeadMsg);

        _accelGyroFixedMsg = std::make_shared<AccelerometerGyroMessage>();
        VPRptr->GetMgrPtr()->AddPublishSubscribeMessage("AccelerometerGyroFixedMsg", _accelGyroFixedMsg);

        _headOrientationCalDataMsg = std::make_shared<HeadOrientationCalDataMessage>();
        VPRptr->GetMgrPtr()->AddPublishSubscribeMessage("HeadOrientationCalDataMsg", _headOrientationCalDataMsg);

        _CalState = CameraCalibrationState_e::CCalState_Reset;

        _calDataDirectoryName = VPRptr->GetConfig()->GetConfigStringValue("CameraCalDataDirectory", "CameraCalData");
        _calDataFileName = "HeadOrientationCalData";

        SendTrackHeadDataOut = VPRptr->GetConfig()->GetConfigBoolValue("HeadTrackingParameters.SendTrackHeadDataOut", false);

        _numberOfCalImages = 0;

        _glyphModelDirectoryName = VPRptr->GetConfig()->GetConfigStringValue("HeadTrackingParameters.GlyphModelsDirectory", "GlyphModels");
        _NoOfGlyphModels = VPRptr->GetConfig()->GetConfigIntValue("HeadTrackingParameters.NumberOfGlyphModels", 1);
        _NoOfGlyphModels = _NoOfGlyphModels < 1 ? 1 : _NoOfGlyphModels > MAXNUMBERGLYPHMODELS ? MAXNUMBERGLYPHMODELS : _NoOfGlyphModels;

        _selectedGlyphModel = 0;
        for(int i = 0; i < _NoOfGlyphModels; i++)
        {
            std::ostringstream cfgParm;

            cfgParm.str(std::string());  //Clear the string stream
            cfgParm << "HeadTrackingParameters.GlyphModelFilename_" << (i + 1);
            _glyphModelFileNames[i] = VPRptr->GetConfig()->GetConfigStringValue(cfgParm.str(), "GlyphModelXLHelmet.glf");

            cfgParm.str(std::string());  //Clear the string stream
            cfgParm << "HeadTrackingParameters.GlyphScale_" << (i + 1);
            _glyphScaleVals[i] = VPRptr->GetConfig()->GetConfigIntValue(cfgParm.str(), 30);
        }

    }

    HeadOrientationCalProcessor::~HeadOrientationCalProcessor()
    {
        _trackHeadProcess.Close();
    }

    std::string HeadOrientationCalProcessor::getGlyphModelFilename(int modelIdx)
    {
        if( modelIdx < 0 || modelIdx >= _NoOfGlyphModels)
            modelIdx = 0;

        boost::filesystem::path filename(_glyphModelDirectoryName);
        filename /= _glyphModelFileNames[modelIdx];
        return filename.string();
    }

    void HeadOrientationCalProcessor::Intialize()
    {
//        ReadHeadModelFromConfig(VPRptr->GetConfig(), _headModelData);


        _headOrientationCalFilename = VPRptr->GetConfig()->GetConfigStringValue("HeadTrackingParameters.HeadOrientationCalFilename", "HeadOrientationCal.ini");
        _htParamsFilename = VPRptr->GetConfig()->GetConfigStringValue("HeadTrackingParameters.HTParametersFilename", "HeadTrackingConfig.ini");

        _headTrackingControlMsg->HeadTrackingParameters.SetDefaults();
        //If there is a Config file... read parameters from the config file
        try
        {
            ReadHeadTrackingParametersFromIniFile(_htParamsFilename, _headTrackingControlMsg->HeadTrackingParameters);

            string glyphFilename = getGlyphModelFilename(_selectedGlyphModel);
            int scale = _glyphScaleVals[_selectedGlyphModel];
            if( !_headModelData.load(glyphFilename, scale) )
            {
                LOGERROR("HeadOrientationCalProcessor Error loading Glyph Model file: " << glyphFilename);
            }
        }
        catch (exception &e)
        {
            LOGWARN("Could not read HeadTrackingConfig.ini Config")
            ReadHeadTrackingParametersFromConfig(VPRptr->GetConfig(), _headTrackingControlMsg->HeadTrackingParameters);
        }

        _headTrackingControlMsg->HeadTrackingImageDisplayType = HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector;
        _headTrackingControlMsg->PostMessage();  //Make sure the rest of videre has these values if needed.

        if(_trackHeadProcess.Initialize(VPRptr->CameraCalData, _headModelData,
                                        _headTrackingControlMsg->HeadTrackingParameters))
        {
            LOGERROR("HeadOrientationCalProcessor Initialize Error.")
        }
        _trackHeadProcess.SetHeadTrackingImageDisplayType(_headTrackingControlMsg->HeadTrackingImageDisplayType);
        _trackHeadProcess.SetUseGPU(false);

        _numberOfCalImages = 0;

        _headOrientationCalData.Clear();
    }


    //Reset the Vision Processor.
    //Reset must return all resources obtained during operation...
    //such a image messages, target result messages and the like.
    void HeadOrientationCalProcessor::Reset()
    {
        if(_currImgMsgPtr != nullptr)
        {
            VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
            _currImgMsgPtr = nullptr;
        }

        //Release Track Head Objects
        _trackHeadProcess.Close();

        _numberOfCalImages = 0;

        _CalState = CameraCalibrationState_e::CCalState_Reset;
        VPRptr->CameraCalStatusMsg->CameraCalibrationState = CameraCalibrationState_e::CCalState_Reset;
        VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
        VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "";
        VPRptr->CameraCalStatusMsg->PostMessage();
    }

    void HeadOrientationCalProcessor::WriteCalDataToFile()
    {
        boost::filesystem::path filename(_calDataDirectoryName);
        filename /= VidereFileUtils::AddOrReplaceFileExtention(_calDataFileName, CAMERA_CAL_FILE_EXT);
        //WriteCameraCalibrationToIniFile(filename.c_str(), _cameraCalData);
    }

    //This is the primary call for running the Vision Processor
    //each time through the Manager's loop.
    VPProcessResult_e HeadOrientationCalProcessor::ExecuteUnitOfWork()
    {
        Quaternion_t q1, q2;
        XYZCoord_t xyzVec;
        XYZCoord_t xyzVec2;
        double rx, theta;
        VPProcessResult_e result = VPProcessResult_e::VPPR_none;
        ImagePlusMetadataMessage *imgPMetadataMsg1Ptr = nullptr;
        ImageProcTargetInfoResultsMessage *tgtMsgPtr;
        Rabit::RabitMessage *tmpMsgPtr;
        std::ostringstream msgBuf;
        VPRptr->StreamRecordControlMsg->FetchMessage();
        bool calCtrlChanged = VPRptr->CameraCalControlMsg->FetchMessage();

        if(_headTrackingControlMsg->FetchMessage())
        {
            VPRptr->ImageProcessControlMsg->FetchMessage();
            _trackHeadProcess.SetHeadTrackingParameters(_headTrackingControlMsg->HeadTrackingParameters);
            _trackHeadProcess.SetHeadTrackingImageDisplayType(_headTrackingControlMsg->HeadTrackingImageDisplayType);
            //The GPU Process does not work as well ... so force this to be false.
            //_trackHeadProcess.SetUseGPU(VPRptr->ImageProcessControlMsg->GPUProcessingEnabled);
            _trackHeadProcess.SetUseGPU(false);
            try
            {
                WriteHeadTrackingParametersToIniFile(_htParamsFilename,
                                                     _headTrackingControlMsg->HeadTrackingParameters);
            }
            catch (exception &e)
            {
                LOGWARN("Could not write HeadTrackingConfig.ini Config")
                ReadHeadTrackingParametersFromConfig(VPRptr->GetConfig(), _headTrackingControlMsg->HeadTrackingParameters);
            }

            if( _headTrackingControlMsg->GlyphModelIndex != _selectedGlyphModel)
            {
                int gmIdx = _headTrackingControlMsg->GlyphModelIndex;
                gmIdx = gmIdx >= _NoOfGlyphModels - 1 ? _NoOfGlyphModels - 1 : gmIdx;
                gmIdx = gmIdx < 0 ? 0 : gmIdx;
                string glyphFilename = getGlyphModelFilename(gmIdx);
                int scale = _glyphScaleVals[gmIdx];
                if( !_headModelData.load(glyphFilename, scale) )
                {
                    LOGERROR("HeadTrackingProcessor Error loading Glyph Model file: " << glyphFilename);
                }
                _selectedGlyphModel = gmIdx;
                _headTrackingControlMsg->GlyphModelIndex = gmIdx;
            }

        }

        try
        {
            switch(_CalState)
            {
                case CameraCalibrationState_e::CCalState_Reset:
                    _capturedImageOk = false;
                    _capturedImageSentForView = false;
                    _numberOfCalImages = 0;
                    _numberSendCapturedImageTries = 0;
                    if(_currImgMsgPtr != nullptr)
                    {
                        VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
                        _currImgMsgPtr = nullptr;
                    }
                    VPRptr->AddEmptyImageMsgsToQueue();
                    _CalState = CameraCalibrationState_e::CCalState_StreamImages;
                    VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                    VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                    VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Head Down Cal";
                    VPRptr->CameraCalStatusMsg->PostMessage();
                    break;

                case CameraCalibrationState_e::CCalState_WaitForStart:
                    switch(VPRptr->CameraCalControlMsg->CameraCalCmd)
                    {
                        case CameraCalCmd_e::CCalCmd_Reset:
                            _CalState = CameraCalibrationState_e::CCalState_Reset;
                            break;
                        case CameraCalCmd_e::CCalCmd_ClearImageSet:
                            _CalState = CameraCalibrationState_e::CCalState_Reset;
                            break;
                        case CameraCalCmd_e::CCalCmd_StreamImages:
                            _CalState = CameraCalibrationState_e::CCalState_StreamImages;
                            VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                            VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                            VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "";
                            VPRptr->CameraCalStatusMsg->PostMessage();
                            break;
                        case CameraCalCmd_e::CCalCmd_RunCalProcess:
                            if(_numberOfCalImages >= 2)
                            {
                                _CalState = CameraCalibrationState_e::CCalState_CalProcess;
                                VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                                VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                                VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Compute Rotation Quaternions";
                                VPRptr->CameraCalStatusMsg->PostMessage();
                            }
                            else
                            {
                                VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                                VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                                VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Need 2 Head Orientations to calibrate";
                                VPRptr->CameraCalStatusMsg->PostMessage();
                            }
                            break;
                    }
                    //Once we have processed the Camera Cal Command.. set to null so we
                    //don't end up in a loop reprocessing an old command.
                    VPRptr->CameraCalControlMsg->CameraCalCmd = CameraCalCmd_e::CCalCmd_NullCmd;
                    break;

                case CameraCalibrationState_e::CCalState_StreamImages:
                    _capturedImageOk = false;
                    _capturedImageSentForView = false;
                    _numberSendCapturedImageTries = 0;
                    if( VPRptr->CameraCalControlMsg->CameraCalCmd == CameraCalCmd_e::CCalCmd_Reset)
                    {
                        _CalState = CameraCalibrationState_e::CCalState_Reset;
                        //Once we have processed the Camera Cal Command.. set to null so we
                        //don't end up in a loop reprocessing an old command.
                        VPRptr->CameraCalControlMsg->CameraCalCmd = CameraCalCmd_e::CCalCmd_NullCmd;
                    }
                    else if ( VPRptr->CameraCalControlMsg->CameraCalCmd == CameraCalCmd_e::CCalCmd_ClearImageSet)
                    {
                        _CalState = CameraCalibrationState_e::CCalState_Reset;
                        //Once we have processed the Camera Cal Command.. set to null so we
                        //don't end up in a loop reprocessing an old command.
                        VPRptr->CameraCalControlMsg->CameraCalCmd = CameraCalCmd_e::CCalCmd_NullCmd;
                    }
                    else if ( VPRptr->CameraCalControlMsg->CameraCalCmd == CameraCalCmd_e::CCalCmd_RunCalProcess)
                    {
                        if(_numberOfCalImages >= 2)
                        {
                            _CalState = CameraCalibrationState_e::CCalState_CalProcess;
                            VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                            VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                            VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Starting Calibraion Process";
                            VPRptr->CameraCalStatusMsg->PostMessage();
                        }
                        else
                        {
                            VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                            VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                            VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Need 2 Head Orientations to calibrate";
                            VPRptr->CameraCalStatusMsg->PostMessage();
                        }
                        //Once we have processed the Camera Cal Command.. set to null so we
                        //don't end up in a loop reprocessing an old command.
                        VPRptr->CameraCalControlMsg->CameraCalCmd = CameraCalCmd_e::CCalCmd_NullCmd;
                    }
                    else
                    {
                        VPRptr->AddEmptyImageMsgsToQueue();
                        _currImgMsgPtr = VPRptr->GetNextIncomingImagePlusMetadataMessage();
                        if (_currImgMsgPtr != nullptr)
                        {

                            _trackHeadOrientationMsg->TrackHeadOrientationData = _trackHeadProcess.TrackHeadPostion(_currImgMsgPtr->ImageFrame);
                            _trackHeadOrientationMsg->ImageNumber = _currImgMsgPtr->ImageNumber;
                            _trackHeadOrientationMsg->ImageCaptureTimeStampSec = _currImgMsgPtr->ImageCaptureTimeStampSec;
                            _trackHeadOrientationMsg->PostMessage();

                            if( SendTrackHeadDataOut )
                            {
                                std::shared_ptr<TrackHeadOrientationMessage> htOutMsg;
                                htOutMsg = std::make_shared<TrackHeadOrientationMessage>();
                                htOutMsg->CopyMessage(_trackHeadOrientationMsg.get());
                                auto rmsgPtr = dynamic_pointer_cast<Rabit::RabitMessage, TrackHeadOrientationMessage>(htOutMsg);
                                VPRptr->GetMgrPtr()->AddMessageToQueue("ZMQPublishOutMsgQueue", rmsgPtr);
                            }

                            //Send a copy of the Image Plus Metadata to the Stream Record Manager.
                            if (VPRptr->StreamRecordEmptyIPMDMsgQueue->GetMessage(tmpMsgPtr))
                            {
                                imgPMetadataMsg1Ptr = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
                                if(PostProcessImageToBeSentOut(_currImgMsgPtr, imgPMetadataMsg1Ptr))
                                {
                                    imgPMetadataMsg1Ptr->CopyMessage(_currImgMsgPtr);
                                }

                                //Send the message back to the Stream Manager even if it is bad,
                                //otherwise we will run out of the messages.
                                VPRptr->StreamRecordRxIPMDMsgQueue->AddMessage(imgPMetadataMsg1Ptr);
                            }
                            if( VPRptr->CameraCalControlMsg->CameraCalCmd == CameraCalCmd_e::CCalCmd_CaptureImage)
                            {
                                //The current image is kept in the _currImgMsgPtr
                                _CalState = CameraCalibrationState_e::CCalState_ImageValidate;
                                VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                                VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                                VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Image Captured: Process Image";
                                VPRptr->CameraCalStatusMsg->PostMessage();

                                //Capture the Head Accelerometer reading
                                _accelGyroHeadMsg->FetchMessage();
                                _capturedHeadAccelOrientation = _accelGyroHeadMsg->AccelerationRates;
                                _capturedFixedAccelOrientation = _accelGyroFixedMsg->AccelerationRates;
                                //Once we have processed the Camera Cal Command.. set to null so we
                                //don't end up in a loop reprocessing an old command.
                                VPRptr->CameraCalControlMsg->CameraCalCmd = CameraCalCmd_e::CCalCmd_NullCmd;
                            }
                            else
                            {
                                //Now return our ImagePlusMetadataMessage back to the pool to be used again.
                                VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
                                _currImgMsgPtr = nullptr;
                            }
                        }
                    }
                    break;

                case CameraCalibrationState_e::CCalState_ImageValidate:
                    //Validate Image and send validation info to user.
                    if(_currImgMsgPtr != nullptr)
                    {
                        //This is redundant... but re-process the captured image and send the
                        //processed image out so that the user can verify the image.
                        _trackHeadOrientationMsg->TrackHeadOrientationData = _trackHeadProcess.TrackHeadPostion(_currImgMsgPtr->ImageFrame);
                        _trackHeadOrientationMsg->ImageNumber = _currImgMsgPtr->ImageNumber;
                        _trackHeadOrientationMsg->ImageCaptureTimeStampSec = _currImgMsgPtr->ImageCaptureTimeStampSec;
                        _trackHeadOrientationMsg->PostMessage();
                        _capturedImageOk = _trackHeadOrientationMsg->TrackHeadOrientationData.IsDataValid;
                        _capturedImageOk = true;
                        //_capturedImageOk = _cameraCal2DObjects.CheckChessBoardImage(_currImgMsgPtr->ImageFrame);
                        //Send a copy of the Image Plus Metadata to the Stream Record Manager.
                        if (VPRptr->StreamRecordEmptyIPMDMsgQueue->GetMessage(tmpMsgPtr))
                        {
                            //Send the image out to be viewed by the user.
                            imgPMetadataMsg1Ptr = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
                            if(PostProcessImageToBeSentOut(_currImgMsgPtr, imgPMetadataMsg1Ptr))
                            {
                                imgPMetadataMsg1Ptr->CopyMessage(_currImgMsgPtr);
                            }
                            imgPMetadataMsg1Ptr->ForceTxImage = true;
                            VPRptr->StreamRecordRxIPMDMsgQueue->AddMessage(imgPMetadataMsg1Ptr);
                            _CalState = CameraCalibrationState_e::CCalState_ImageCapturedWait;
                            VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                            VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                            if( _capturedImageOk)
                            {
                                VPRptr->CameraCalStatusMsg->ImageOk = true;
                                VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Image Ok: Select/Reject Image";
                            }
                            else
                            {
                                VPRptr->CameraCalStatusMsg->ImageOk = false;
                                VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Image Bad: Reject Image";
                            }
                            VPRptr->CameraCalStatusMsg->PostMessage();
                        }
                        else if( ++_numberSendCapturedImageTries > 3)
                        {
                            LOGWARN("Head Orientation Calibration Processor: Cannot get Empty IPM from Queue, Continue.");
                            //Go To Image Accept anyway.
                            _CalState = CameraCalibrationState_e::CCalState_ImageCapturedWait;
                            VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                            VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                            if( _capturedImageOk)
                                VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Image Ok: Select/Reject Image";
                            else
                                VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Image Bad: Reject Image";
                            VPRptr->CameraCalStatusMsg->PostMessage();
                        }
                        else
                        {
                            //Go around the loop for three tries.
                            LOGWARN("Camera Calibration Processor: Cannot get Empty IPM from Queue... will try again.");
                        }
                    }
                    else
                    {
                        LOGERROR("Camera Calibration Processor: In ImageValidate without an Image.");
                        _CalState = CameraCalibrationState_e::CCalState_StreamImages;
                        VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                        VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                        VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Error: No Image... try again.";
                        VPRptr->CameraCalStatusMsg->ImageOk = false;
                        VPRptr->CameraCalStatusMsg->PostMessage();
                    }

                    break;

                case CameraCalibrationState_e::CCalState_ImageCapturedWait:
                    if( VPRptr->CameraCalControlMsg->CameraCalCmd == CameraCalCmd_e::CCalCmd_Reset)
                    {
                        VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
                        _currImgMsgPtr = nullptr;
                        _CalState = CameraCalibrationState_e::CCalState_Reset;
                    }
                    else if ( VPRptr->CameraCalControlMsg->CameraCalCmd == CameraCalCmd_e::CCalCmd_SetImageOk)
                    {
                        //Capture the Orientation Info.
                        if(_numberOfCalImages < 2)
                        {
                            CapturedHeadOrientations[_numberOfCalImages] =
                                    _trackHeadOrientationMsg->TrackHeadOrientationData.HeadOrientationQuaternion;
                        }

                        //Now return our ImagePlusMetadataMessage back to the pool to be used again.
                        VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
                        _currImgMsgPtr = nullptr;
                        VPRptr->AddEmptyImageMsgsToQueue();
                        bool storedError = false;

                        ++_numberOfCalImages;
                        if( _numberOfCalImages < 2 )
                        {
                            _CalState = CameraCalibrationState_e::CCalState_StreamImages;
                            VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                            VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                            VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Head Forward Cal";
                            VPRptr->CameraCalStatusMsg->PostMessage();
                        }
                        else
                        {
                            //We are done... go the computation state
                            _CalState = CameraCalibrationState_e::CCalState_CalProcess;
                            VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                            VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                            VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Done";
                            VPRptr->CameraCalStatusMsg->PostMessage();
                        }
                    }
                    else if ( VPRptr->CameraCalControlMsg->CameraCalCmd == CameraCalCmd_e::CCalCmd_RejectImage)
                    {
                        //Now return our ImagePlusMetadataMessage back to the pool to be used again.
                        VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
                        _currImgMsgPtr = nullptr;
                        VPRptr->AddEmptyImageMsgsToQueue();
                        _CalState = CameraCalibrationState_e::CCalState_StreamImages;
                        VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                        VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                        VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "";
                        VPRptr->CameraCalStatusMsg->PostMessage();
                    }
                    //Once we have processed the Camera Cal Command.. set to null so we
                    //don't end up in a loop reprocessing an old command.
                    VPRptr->CameraCalControlMsg->CameraCalCmd = CameraCalCmd_e::CCalCmd_NullCmd;
                    break;

                case CameraCalibrationState_e::CCalState_CalProcess:
                    //Calibration process of Orientation data and write to calibration files
                    q1.SetQuaternion(0.5, 0.5, 0.5, 0.5);
                    q2 = q1 * CapturedHeadOrientations[0].getCongugate();
                    _headOrientationCalData.CameraToCarQ = q2;
                    _headOrientationCalData.HeadToModelQ = CapturedHeadOrientations[1].getCongugate() * q2.getCongugate();

                    //Compute the Gyro - to Head Quaternion based upon the Head Accelerometer Reading
                    xyzVec = _capturedHeadAccelOrientation.NormalizedVector();
                    xyzVec2 = _capturedFixedAccelOrientation.NormalizedVector();

                    //*************** Old process assumes head gyro centered on top of helmet *****
                    theta = 0.5 * asin(-xyzVec.x);
                    q1.SetQuaternion(0, 0, 0, 1.0);
                    q2.SetQuaternion(cos(theta), 0, -sin(theta), 0);
                    _headOrientationCalData.GyroToHeadQ = q2 * q1;
                    LOGINFO("HeadOrientationCalData.GyroToHead sf:" << _headOrientationCalData.GyroToHeadQ.qScale
                          << " x:" << _headOrientationCalData.GyroToHeadQ.qVec.x
                          << " y:" << _headOrientationCalData.GyroToHeadQ.qVec.y
                          << " z:" << _headOrientationCalData.GyroToHeadQ.qVec.z );
                     //*****************************************************************************/

                    //The Fixed/car accel values are not currently being captured
                    //so we assume a straight down vec.
                    xyzVec2.SetXYZCoord(0.0, 0.0, 1.0);
                    q2 = Quaternion_t::Vec1ToVec2RotationQuaternion(xyzVec2, xyzVec);
                    LOGINFO("HeadOrientationCalData.q2 sf:" << q2.qScale
                                                                    << " x:" << q2.qVec.x
                                                                    << " y:" << q2.qVec.y
                                                                    << " z:" << q2.qVec.z );
                    q1.SetQuaternion(0, 0, 0, 1.0);
                    _headOrientationCalData.GyroToHeadQ = q2 * q1;
                    WriteHeadOrientationCalDataToIniFile(_headOrientationCalFilename, _headOrientationCalData);
                    LOGINFO("HeadOrientationCalData.GyroToHead sf:" << _headOrientationCalData.GyroToHeadQ.qScale
                                                                    << " x:" << _headOrientationCalData.GyroToHeadQ.qVec.x
                                                                    << " y:" << _headOrientationCalData.GyroToHeadQ.qVec.y
                                                                    << " z:" << _headOrientationCalData.GyroToHeadQ.qVec.z );

                    _headOrientationCalDataMsg->CalData = _headOrientationCalData;
                    _headOrientationCalDataMsg->PostMessage();

                    _CalState = CameraCalibrationState_e::CCalState_CalComplete;
                    VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                    VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                    VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Done";
                    VPRptr->CameraCalStatusMsg->PostMessage();
                    break;

                case CameraCalibrationState_e::CCalState_CalComplete:
                    if( VPRptr->CameraCalControlMsg->CameraCalCmd == CameraCalCmd_e::CCalCmd_Reset)
                    {
                        _CalState = CameraCalibrationState_e::CCalState_Reset;
                    }
                    //Once we have processed the Camera Cal Command.. set to null so we
                    //don't end up in a loop reprocessing an old command.
                    VPRptr->CameraCalControlMsg->CameraCalCmd = CameraCalCmd_e::CCalCmd_NullCmd;
                    break;

                case CameraCalibrationState_e::CCalState_CalError:
                    if( VPRptr->CameraCalControlMsg->CameraCalCmd == CameraCalCmd_e::CCalCmd_Reset)
                    {
                        _CalState = CameraCalibrationState_e::CCalState_Reset;
                    }
                    //Once we have processed the Camera Cal Command.. set to null so we
                    //don't end up in a loop reprocessing an old command.
                    VPRptr->CameraCalControlMsg->CameraCalCmd = CameraCalCmd_e::CCalCmd_NullCmd;
                    break;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("Head Orientation Calibration Processor: Exception: " << e.what());
        }

        return result;
    }


    bool HeadOrientationCalProcessor::PostProcessImageToBeSentOut(ImagePlusMetadataMessage *inputMsg,
                                                            ImagePlusMetadataMessage *outputMsg)
    {
        bool error = true;
        cv::Mat outpImage;
        if (inputMsg != nullptr && outputMsg != nullptr)
        {
            try
            {
                _headTrackingControlMsg->FetchMessage();
                switch(_headTrackingControlMsg->HeadTrackingImageDisplayType)
                {
                    case HeadTrackingImageDisplayType_e::HTID_None:
                        outputMsg->CopyMessage(inputMsg);
                        break;

                    case HeadTrackingImageDisplayType_e::HTID_HighLightGlyphs:
                        if( _trackHeadProcess.GetIntermediateImage(0, outputMsg->ImageFrame))
                        {
                            outputMsg->CopyMessageWithoutImage(inputMsg);
                        }
                        else
                        {
                            outputMsg->CopyMessage(inputMsg);
                        }
                        break;
                    case HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector:
                        if( _trackHeadProcess.GetIntermediateImage(1, outputMsg->ImageFrame))
                        {
                            outputMsg->CopyMessageWithoutImage(inputMsg);
                        }
                        else
                        {
                            outputMsg->CopyMessage(inputMsg);
                        }
                        break;
                }
                error = false;
            }
            catch (std::exception &e)
            {
                LOGERROR("HeadTrackingProcessor:PostProcessImageToBeSentOut: Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }


}