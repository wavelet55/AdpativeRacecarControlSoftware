/* ****************************************************************
 * Standard Target Detector Vision Processor
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Feb. 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#include "CameraCalibrationProcessor.h"
#include "FileUtils.h"
#include <opencv2/core/core.hpp>
#include "CameraCalibration/CameraCalReaderWriter.h"

using namespace CameraCalibrationNS;

namespace videre
{

    CameraCalibrationProcessor::CameraCalibrationProcessor(VisionProcessResources* vpResources)
        : VisionProcessorAbstract(vpResources),
          _JpgFileHandler(90)
    {
        _CalType = CameraCalibrationType_e::CameraCal_2DPlaneCheckerBoard;
        _CalState = CameraCalibrationState_e::CCalState_Reset;
        _calImageDirectoryName = VPRptr->GetConfig()->GetConfigStringValue("CameraCalImageDirectory", "CameraCalImages");
        _calImageBaseName = VPRptr->GetConfig()->GetConfigStringValue("CameraCalImageFileBaseName", "CalImage_");
        _imageFormatType = ImageFormatType_e::ImgFType_JPEG;
        _calDataDirectoryName = VPRptr->GetConfig()->GetConfigStringValue("CameraCalDataDirectory", "CameraCalData");
        _numberOfCalImages = 0;
        _calDataFileName = "CalData";
    }

    CameraCalibrationProcessor::~CameraCalibrationProcessor()
    {

    }


    //Reset the Vision Processor.
    //Reset must return all resources obtained during operation...
    //such a image messages, target result messages and the like.
    void CameraCalibrationProcessor::Reset()
    {
        if(_currImgMsgPtr != nullptr)
        {
            VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
            _currImgMsgPtr = nullptr;
        }

        _JpgFileHandler.Reset();

        _CalState = CameraCalibrationState_e::CCalState_Reset;
        VPRptr->CameraCalStatusMsg->CameraCalibrationState = CameraCalibrationState_e::CCalState_Reset;
        VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
        VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "";
        VPRptr->CameraCalStatusMsg->PostMessage();
    }

    std::string CameraCalibrationProcessor::CreateImageFilename(int fileIdx)
    {
        boost::filesystem::path filePlusDirName(_calImageDirectoryName);
        string filename = VidereFileUtils::AddIndexToFilename(_calImageBaseName,
                                                              fileIdx, 3,
                                                              VidereFileUtils::GetImageFileExtention(_imageFormatType));
        filePlusDirName /= filename;
        return filePlusDirName.c_str();
    }

    void CameraCalibrationProcessor::WriteCalDataToFile()
    {
        boost::filesystem::path filename(_calDataDirectoryName);
        filename /= VidereFileUtils::AddOrReplaceFileExtention(_calDataFileName, CAMERA_CAL_FILE_EXT);
        WriteCameraCalibrationToIniFile(filename.c_str(), _cameraCalData);
    }

    void CameraCalibrationProcessor::ReadCameraMountCorrectionParameters(const shared_ptr<CameraCalCommandMessage> calDataMsg)
    {
        _cameraCalData.UseCameraMountingCorrection = true;
        _cameraCalData.SetYawCorrectionDegrees(calDataMsg->YawCorrectionDegrees);
        _cameraCalData.SetPitchCorrectionDegrees(calDataMsg->PitchCorrectionDegrees);
        _cameraCalData.SetRollCorrectionDegrees(calDataMsg->RollCorrectionDegrees);
        _cameraCalData.SetDelXCorrectionCentiMeters(calDataMsg->DelXCorrectionCentiMeters);
        _cameraCalData.SetDelYCorrectionCentiMeters(calDataMsg->DelYCorrectionCentiMeters);
        _cameraCalData.SetDelZCorrectionCentiMeters(calDataMsg->DelZCorrectionCentiMeters);
        _cameraCalData.GenerateRotationXlationCalFromCameraMountingCorrection();
    }

    //This is the primary call for running the Vision Processor
    //each time through the Manager's loop.
    VPProcessResult_e CameraCalibrationProcessor::ExecuteUnitOfWork()
    {
        VPProcessResult_e result = VPProcessResult_e::VPPR_none;
        ImagePlusMetadataMessage *imgPMetadataMsg1Ptr = nullptr;
        ImageProcTargetInfoResultsMessage *tgtMsgPtr;
        Rabit::RabitMessage *tmpMsgPtr;
        std::ostringstream msgBuf;
        VPRptr->StreamRecordControlMsg->FetchMessage();
        bool calCtrlChanged = VPRptr->CameraCalControlMsg->FetchMessage();
        if(calCtrlChanged)
        {
            //cout << "CameraCalControlMsg" << VPRptr->CameraCalControlMsg->CameraCalCmd << endl;
            ReadCameraMountCorrectionParameters(VPRptr->CameraCalControlMsg);
        }

        try
        {
            switch(_CalState)
            {
                case CameraCalibrationState_e::CCalState_Reset:
                    _capturedImageOk = false;
                    _capturedImageSentForView = false;
                    _numberSendCapturedImageTries = 0;
                    if(_currImgMsgPtr != nullptr)
                    {
                        VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
                        _currImgMsgPtr = nullptr;
                    }
                    VPRptr->AddEmptyImageMsgsToQueue();
                    //Read Cal Images Directory to find the current number of images
                    //available.
                    VidereFileUtils::CreateDirectory(_calImageDirectoryName);
                    //Get a list of the images in the Cal Image Directory.
                    _numberOfCalImages = VidereFileUtils::GetListFilesInDirectory(&_listOfCalImageFiles,
                                                                                  _calImageDirectoryName,
                                                VidereFileUtils::GetImageFileExtention(_imageFormatType),
                                                "", true);

                    //Ensure we have a directory setup for the Calibration data to be stored.
                    VidereFileUtils::CreateDirectory(VPRptr->GetConfig()->GetConfigStringValue("CameraCalDataDirectory", "CameraCalData"));

                    //Set Calibration Parameters
                    //ToDo: Get parameters from Camera Control Msg.
                    _cameraCal2DObjects.CalType = VPRptr->CameraCalControlMsg->CameraCalibrationType;
                    _cameraCal2DObjects.Set_NumberOfObjects_X_Axis(VPRptr->CameraCalControlMsg->NumberOfCols);
                    _cameraCal2DObjects.Set_NumberOfObjects_Y_Axis(VPRptr->CameraCalControlMsg->NumberOfRows);
                    _cameraCal2DObjects.Set_ChessBoardSquareSizeMilliMeters(VPRptr->CameraCalControlMsg->SquareSizeMilliMeters);

                    ReadCameraMountCorrectionParameters(VPRptr->CameraCalControlMsg);

                    if( VPRptr->CameraCalControlMsg->CameraCalBaseFilename.length() > 1)
                    {
                        _calDataFileName = VPRptr->CameraCalControlMsg->CameraCalBaseFilename;
                    }
                    _CalState = CameraCalibrationState_e::CCalState_WaitForStart;
                    VPRptr->CameraCalStatusMsg->CameraCalibrationState = CameraCalibrationState_e::CCalState_WaitForStart;
                    VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                    VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "";
                    VPRptr->CameraCalStatusMsg->ImageOk = false;
                    VPRptr->CameraCalStatusMsg->PostMessage();
                    break;

                case CameraCalibrationState_e::CCalState_WaitForStart:
                    if(calCtrlChanged)
                    {
                        _cameraCal2DObjects.CalType = VPRptr->CameraCalControlMsg->CameraCalibrationType;
                        _cameraCal2DObjects.Set_NumberOfObjects_X_Axis(VPRptr->CameraCalControlMsg->NumberOfCols);
                        _cameraCal2DObjects.Set_NumberOfObjects_Y_Axis(VPRptr->CameraCalControlMsg->NumberOfRows);
                        _cameraCal2DObjects.Set_ChessBoardSquareSizeMilliMeters(VPRptr->CameraCalControlMsg->SquareSizeMilliMeters);
                        if( VPRptr->CameraCalControlMsg->CameraCalBaseFilename.length() > 1)
                        {
                            _calDataFileName = VPRptr->CameraCalControlMsg->CameraCalBaseFilename;
                        }
                    }
                    switch(VPRptr->CameraCalControlMsg->CameraCalCmd)
                    {
                        case CameraCalCmd_e::CCalCmd_Reset:
                            _CalState = CameraCalibrationState_e::CCalState_Reset;
                            break;
                        case CameraCalCmd_e::CCalCmd_ClearImageSet:
                            VidereFileUtils::DeleteDirectoryAndAllFiles(_calImageDirectoryName);
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
                            if(_numberOfCalImages > 2)
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
                                VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Need atleast 3 Images to run calibraion process";
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
                        VidereFileUtils::DeleteDirectoryAndAllFiles(_calImageDirectoryName);
                        _CalState = CameraCalibrationState_e::CCalState_Reset;
                        //Once we have processed the Camera Cal Command.. set to null so we
                        //don't end up in a loop reprocessing an old command.
                        VPRptr->CameraCalControlMsg->CameraCalCmd = CameraCalCmd_e::CCalCmd_NullCmd;
                    }
                    else if ( VPRptr->CameraCalControlMsg->CameraCalCmd == CameraCalCmd_e::CCalCmd_RunCalProcess)
                    {
                        if(_numberOfCalImages > 2)
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
                            VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Need atleast 3 Images to run calibraion process";
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
                            //Send a copy of the Image Plus Metadata to the Stream Record Manager.
                            if (VPRptr->StreamRecordEmptyIPMDMsgQueue->GetMessage(tmpMsgPtr))
                            {
                                imgPMetadataMsg1Ptr = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
                                imgPMetadataMsg1Ptr->CopyMessage(_currImgMsgPtr);
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
                        _capturedImageOk = _cameraCal2DObjects.CheckChessBoardImage(_currImgMsgPtr->ImageFrame);
                            //Send a copy of the Image Plus Metadata to the Stream Record Manager.
                        if (VPRptr->StreamRecordEmptyIPMDMsgQueue->GetMessage(tmpMsgPtr))
                        {
                            //Send the image out to be viewed by the user.
                            imgPMetadataMsg1Ptr = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
                            imgPMetadataMsg1Ptr->CopyMessageWithoutImage(_currImgMsgPtr);
                            _cameraCal2DObjects.GrayScaleImg.copyTo(imgPMetadataMsg1Ptr->ImageFrame);
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
                            LOGWARN("Camera Calibration Processor: Cannot get Empty IPM from Queue, Continue.");
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
                        //Save Image
                        string fn = CreateImageFilename(_numberOfCalImages + 1);
                        bool storedError = !_JpgFileHandler.CompressAndStoreImage(_currImgMsgPtr->ImageFrame, fn);

                        //Now return our ImagePlusMetadataMessage back to the pool to be used again.
                        VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
                        _currImgMsgPtr = nullptr;
                        VPRptr->AddEmptyImageMsgsToQueue();
                        _CalState = CameraCalibrationState_e::CCalState_StreamImages;

                        if(storedError )
                        {
                            //Error compressing and storing the image... treat as rejected.
                            VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                            VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                            VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Error Storing Image";
                            VPRptr->CameraCalStatusMsg->PostMessage();
                        }
                        else
                        {
                            ++_numberOfCalImages;
                            //Add file to list of files to process.
                            boost::filesystem::path newImageFilename(fn);
                            _listOfCalImageFiles.push_back(newImageFilename);
                            VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                            VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                            VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Image Stored";
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
                    //Calibration process of all images.
                    if(_listOfCalImageFiles.size() > 0)
                    {
                        msgBuf.clear();
                        msgBuf.str("");
                        msgBuf << "Processing Images: " << _listOfCalImageFiles.size();
                        VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = msgBuf.str();
                        VPRptr->CameraCalStatusMsg->PostMessage();
                        int numOfImgsProcced = _cameraCal2DObjects.ProcessCalImages(_listOfCalImageFiles);

                        if (numOfImgsProcced > 3)
                        {
                            msgBuf.clear();
                            msgBuf.str("");
                            msgBuf << "Running calibration on Images: " << numOfImgsProcced;

                            VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = msgBuf.str();
                            VPRptr->CameraCalStatusMsg->PostMessage();
                            numOfImgsProcced = _cameraCal2DObjects.RunCalibration();

                            if(numOfImgsProcced > 0 )
                            {
                                double calError = _cameraCal2DObjects.computeReprojectionErrors();

                                _CalState = CameraCalibrationState_e::CCalState_CalComplete;
                                _cameraCal2DObjects.WriteCalToCameraCalData(_cameraCalData);
                                WriteCalDataToFile();

                                VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                                VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = numOfImgsProcced;

                                msgBuf.clear();
                                msgBuf.str("");
                                msgBuf << "Cal Error: " << calError;

                                VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = msgBuf.str();
                                VPRptr->CameraCalStatusMsg->PostMessage();
                            }
                            else
                            {
                                _CalState = CameraCalibrationState_e::CCalState_CalError;
                                VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                                VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = numOfImgsProcced;
                                VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Calibration Process Error";
                                VPRptr->CameraCalStatusMsg->PostMessage();
                            }
                        }
                        else
                        {
                            _CalState = CameraCalibrationState_e::CCalState_CalError;
                            VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                            VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = numOfImgsProcced;
                            VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Calibration Error... Not enough valid images";
                            VPRptr->CameraCalStatusMsg->PostMessage();
                        }
                    }
                    else
                    {
                        _CalState = CameraCalibrationState_e::CCalState_CalError;
                        VPRptr->CameraCalStatusMsg->CameraCalibrationState = _CalState;
                        VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfCalImages;
                        VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "Calibration Error... Not enough valid images";
                        VPRptr->CameraCalStatusMsg->PostMessage();
                    }
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
            LOGERROR("Camera Calibration Processor: Exception: " << e.what());
        }

       return result;
    }



}