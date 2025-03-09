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


#include "FeatureMatchingProcessor.h"
#include "FileUtils.h"
#include "StdCommon/ImageKeypointsAndFeatures.h"
#include "StdCommon/ImageFeatureMatcher.h"
#include <opencv2/core/core.hpp>

using namespace ImageProcLibsNS;

namespace videre
{

    FeatureMatchingProcessor::FeatureMatchingProcessor(VisionProcessResources* vpResources)
            : VisionProcessorAbstract(vpResources),
              _imageKeyptsFeaturesPrimary(),
              _imageKeyptsFeaturesQuery(),
              _imagePrimary(),
              _imageFeatureMatcher(),
              _JpgFileHandler(90)
    {
        _FMPState = FeatureMatchingState_e::FMPState_Reset;
        _featureImageDirectoryName = VPRptr->GetConfig()->GetConfigStringValue("FeatureMatchingImageDirectory", "FeatureImages");
        _featureImageBaseName = VPRptr->GetConfig()->GetConfigStringValue("FeatureMatchingImageFileBaseName", "FeatureImage_");
        _imageFormatType = ImageFormatType_e::ImgFType_JPEG;
        _featureDataDirectoryName = VPRptr->GetConfig()->GetConfigStringValue("FeatureMatchingDataDirectory", "FeatureMatchingData");
        _numberOfFeatureImages = 0;
        _featureDataFileName = "CalData";
        //Stopwatches for general purpose
        _stopwatch1.reset();
        _stopwatch2.reset();
    }


    FeatureMatchingProcessor::~FeatureMatchingProcessor()
    {

    }


    //Reset the Vision Processor.
    //Reset must return all resources obtained during operation...
    //such a image messages, target result messages and the like.
    void FeatureMatchingProcessor::Reset()
    {
        if(_currImgMsgPtr != nullptr)
        {
            VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
            _currImgMsgPtr = nullptr;
        }

        _JpgFileHandler.Reset();

        _FMPState = FeatureMatchingState_e::FMPState_Reset;
        //VPRptr->CameraCalStatusMsg->CameraCalibrationState = CameraCalibrationState_e::CCalState_Reset;
        //VPRptr->CameraCalStatusMsg->NumberOfImagesCaptured = _numberOfFeatureImages;
        //VPRptr->CameraCalStatusMsg->CameraCalStatusMsg = "";
        //VPRptr->CameraCalStatusMsg->PostMessage();
    }

    std::string FeatureMatchingProcessor::CreateImageFilename(int fileIdx)
    {
        boost::filesystem::path filePlusDirName(_featureImageDirectoryName);
        string filename = VidereFileUtils::AddIndexToFilename(_featureImageBaseName,
                                                              fileIdx, 3,
                                                              VidereFileUtils::GetImageFileExtention(_imageFormatType));
        filePlusDirName /= filename;
        return filePlusDirName.c_str();
    }

    //This is the primary call for running the Vision Processor
    //each time through the Manager's loop.
    VPProcessResult_e FeatureMatchingProcessor::ExecuteUnitOfWork()
    {
        VPProcessResult_e result = VPProcessResult_e::VPPR_none;
        ImagePlusMetadataMessage *imgPMetadataMsg1Ptr = nullptr;
        ImageProcTargetInfoResultsMessage *tgtMsgPtr;
        Rabit::RabitMessage *tmpMsgPtr;
        std::ostringstream msgBuf;
        bool ctrlMsgChanged = VPRptr->FeatureMatchProcCtrlMsg->FetchMessage();
        bool featureMatchStatusMsgChanged = false;
        if(ctrlMsgChanged)
        {
            if(VPRptr->FeatureMatchProcCtrlMsg->FeatureExtractionTypeRoutine
                    == FeatureExtractionTypeRoutine_e::FETR_ORB )
            {
                _imageKeyptsFeaturesPrimary.setFeatureDetectorType(FeatureDetectorType_e::FDT_ORB);
                _imageKeyptsFeaturesQuery.setFeatureDetectorType(FeatureDetectorType_e::FDT_ORB);
            }
            else if(VPRptr->FeatureMatchProcCtrlMsg->FeatureExtractionTypeRoutine
               == FeatureExtractionTypeRoutine_e::FETR_SIFT )
            {
                _imageKeyptsFeaturesPrimary.setFeatureDetectorType(FeatureDetectorType_e::FDT_SIFT);
                _imageKeyptsFeaturesQuery.setFeatureDetectorType(FeatureDetectorType_e::FDT_SIFT);
            }

            if (VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchTypeRoutine == FeatureMatchTypeRoutine_e::FMTR_BruteForce)
            {
                _imageFeatureMatcher.setFeatureMatcherType(FeatureMatcherType_e::FMT_BruteForce);
            }
            else if(VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchTypeRoutine != FeatureMatchTypeRoutine_e::FMTR_FLANN)
            {
                _imageFeatureMatcher.setFeatureMatcherType(FeatureMatcherType_e::FMT_FLANN);
            }

            VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = FeatureMatchingState_e::FMPState_WaitForStart;
            VPRptr->FeatureMatchProcStatusMsg->FeatureExtractionTypeRoutine = VPRptr->FeatureMatchProcCtrlMsg->FeatureExtractionTypeRoutine;
            VPRptr->FeatureMatchProcStatusMsg->FeatureMatchTypeRoutine = VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchTypeRoutine;
            featureMatchStatusMsgChanged = true;
        }

        try
        {
            switch (_FMPState)
            {
                case FeatureMatchingState_e::FMPState_Reset:
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
                    VidereFileUtils::CreateDirectory(_featureImageDirectoryName);
                    //Get a list of the images in the Cal Image Directory.
                    _numberOfFeatureImages = VidereFileUtils::GetListFilesInDirectory(&_listOfFearueImageFiles,
                                                                                      _featureImageDirectoryName,
                                                                                  VidereFileUtils::GetImageFileExtention(_imageFormatType),
                                                                                  "", true);
                    //Ensure we have a directory setup for the Calibration data to be stored.
                    VidereFileUtils::CreateDirectory(_featureDataDirectoryName);

                    VPRptr->FeatureMatchProcStatusMsg->Clear();
                    VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = FeatureMatchingState_e::FMPState_WaitForStart;
                    VPRptr->FeatureMatchProcStatusMsg->FeatureExtractionTypeRoutine = VPRptr->FeatureMatchProcCtrlMsg->FeatureExtractionTypeRoutine;
                    VPRptr->FeatureMatchProcStatusMsg->FeatureMatchTypeRoutine = VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchTypeRoutine;
                    featureMatchStatusMsgChanged = true;

                    //Go to next state
                    _FMPState = FeatureMatchingState_e::FMPState_WaitForStart;
                    break;

                case FeatureMatchingState_e::FMPState_WaitForStart:
                    switch(VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchingProcCmd)
                    {
                        case FeatureMatchingProcCmd_e::FMPCmd_Reset:
                            _FMPState = FeatureMatchingState_e::FMPState_Reset;
                            break;
                        case FeatureMatchingProcCmd_e::FMPCmd_ClearImageSet:
                            VidereFileUtils::DeleteDirectoryAndAllFiles(_featureImageDirectoryName);
                            _FMPState = FeatureMatchingState_e::FMPState_Reset;
                            break;
                        case FeatureMatchingProcCmd_e::FMPCmd_StreamImages:
                            _FMPState = FeatureMatchingState_e::FMPState_StreamImages;
                            VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = _FMPState;
                            VPRptr->FeatureMatchProcStatusMsg->NumberOfImagesCaptured = _numberOfFeatureImages;
                            VPRptr->FeatureMatchProcStatusMsg->StatusMessage = "";
                            featureMatchStatusMsgChanged = true;
                            break;
                        case FeatureMatchingProcCmd_e::FMPCmd_RunImageProcess:
                            if(_numberOfFeatureImages > 0)
                            {
                                _FMPState = FeatureMatchingState_e::FMPState_FMProcess;
                                VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = _FMPState;
                                VPRptr->FeatureMatchProcStatusMsg->NumberOfImagesCaptured = _numberOfFeatureImages;
                                VPRptr->FeatureMatchProcStatusMsg->StatusMessage = "Starting Image Processing";
                                featureMatchStatusMsgChanged = true;
                            }
                            else
                            {
                                VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = _FMPState;
                                VPRptr->FeatureMatchProcStatusMsg->NumberOfImagesCaptured = _numberOfFeatureImages;
                                VPRptr->FeatureMatchProcStatusMsg->StatusMessage = "Need atleast 1 Object Images to run matching process";
                                featureMatchStatusMsgChanged = true;
                            }
                            break;
                    }
                    //Once we have processed the FeatureMatchProcCtrl Command.. set to null so we
                    //don't end up in a loop reprocessing an old command.
                    VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchingProcCmd = FeatureMatchingProcCmd_e::FMPCmd_NullCmd;
                    break;

                case FeatureMatchingState_e::FMPState_StreamImages:
                    _capturedImageOk = false;
                    _capturedImageSentForView = false;
                    _numberSendCapturedImageTries = 0;
                    switch(VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchingProcCmd)
                    {
                        case FeatureMatchingProcCmd_e::FMPCmd_Reset:
                            _FMPState = FeatureMatchingState_e::FMPState_Reset;
                            break;
                        case FeatureMatchingProcCmd_e::FMPCmd_ClearImageSet:
                            VidereFileUtils::DeleteDirectoryAndAllFiles(_featureImageDirectoryName);
                            _FMPState = FeatureMatchingState_e::FMPState_Reset;
                            break;
                        case FeatureMatchingProcCmd_e::FMPCmd_StreamImages:
                            _FMPState = FeatureMatchingState_e::FMPState_StreamImages;
                            VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = _FMPState;
                            VPRptr->FeatureMatchProcStatusMsg->NumberOfImagesCaptured = _numberOfFeatureImages;
                            VPRptr->FeatureMatchProcStatusMsg->StatusMessage = "";
                            featureMatchStatusMsgChanged = true;
                            break;
                        case FeatureMatchingProcCmd_e::FMPCmd_RunImageProcess:
                            if(_numberOfFeatureImages > 0)
                            {
                                _FMPState = FeatureMatchingState_e::FMPState_FMProcess;
                                VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = _FMPState;
                                VPRptr->FeatureMatchProcStatusMsg->NumberOfImagesCaptured = _numberOfFeatureImages;
                                VPRptr->FeatureMatchProcStatusMsg->StatusMessage = "Starting Image Processing";
                                featureMatchStatusMsgChanged = true;
                            }
                            else
                            {
                                VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = _FMPState;
                                VPRptr->FeatureMatchProcStatusMsg->NumberOfImagesCaptured = _numberOfFeatureImages;
                                VPRptr->FeatureMatchProcStatusMsg->StatusMessage = "Need atleast 1 Object Images to run matching process";
                                featureMatchStatusMsgChanged = true;
                            }
                            break;
                        default:
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
                                if(VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchingProcCmd
                                   == FeatureMatchingProcCmd_e::FMPCmd_CaptureImage)
                                {
                                    //The current image is kept in the _currImgMsgPtr
                                    //Keep it and do not check it in.
                                    _FMPState = FeatureMatchingState_e::FMPState_ImageValidate;
                                    VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = _FMPState;
                                    VPRptr->FeatureMatchProcStatusMsg->NumberOfImagesCaptured = _numberOfFeatureImages;
                                    VPRptr->FeatureMatchProcStatusMsg->StatusMessage = "Image Captured. Process and Validate It.";
                                    featureMatchStatusMsgChanged = true;
                                }
                                else
                                {
                                    //Now return our ImagePlusMetadataMessage back to the pool to be used again.
                                    VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
                                    _currImgMsgPtr = nullptr;
                                }
                            }
                            break;
                    }
                    //Once we have processed the FeatureMatchProcCtrl Command.. set to null so we
                    //don't end up in a loop reprocessing an old command.
                    VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchingProcCmd = FeatureMatchingProcCmd_e::FMPCmd_NullCmd;

                    break;

                case FeatureMatchingState_e::FMPState_ImageValidate:
                     //Validate Image and send validation info to user.
                    if(_currImgMsgPtr != nullptr)
                    {
                        ProcessImageFeatures(_currImgMsgPtr);

                        if (VPRptr->StreamRecordEmptyIPMDMsgQueue->GetMessage(tmpMsgPtr))
                        {
                            //Send the image out to be viewed by the user.
                            imgPMetadataMsg1Ptr = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
                            imgPMetadataMsg1Ptr->CopyMessage(_currImgMsgPtr);
                            PostProcessImageToBeSentOut(imgPMetadataMsg1Ptr);
                            imgPMetadataMsg1Ptr->ForceTxImage = true;
                            VPRptr->StreamRecordRxIPMDMsgQueue->AddMessage(imgPMetadataMsg1Ptr);

                            _FMPState = FeatureMatchingState_e::FMPState_ImageCapturedWait;
                            VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = _FMPState;
                            VPRptr->FeatureMatchProcStatusMsg->NumberOfImagesCaptured = _numberOfFeatureImages;
                            VPRptr->FeatureMatchProcStatusMsg->StatusMessage = "Image Captured. Wait for Approval.";
                            featureMatchStatusMsgChanged = true;
                        }
                        else if( ++_numberSendCapturedImageTries > 3)
                        {
                            LOGERROR("Feature Matching Processor: Cannot get Empty IPM from Queue.");
                            _FMPState = FeatureMatchingState_e::FMPState_StreamImages;
                            VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = _FMPState;
                            VPRptr->FeatureMatchProcStatusMsg->NumberOfImagesCaptured = _numberOfFeatureImages;
                            VPRptr->FeatureMatchProcStatusMsg->StatusMessage = "Cannot get Empty IPM from Queue.";
                            featureMatchStatusMsgChanged = true;
                        }
                        else
                        {
                            //Go around the loop for three tries.
                            LOGWARN("Camera Calibration Processor: Cannot get Empty IPM from Queue... will try again.");
                        }
                    }
                    else
                    {
                        //No Image... go back to stream images
                        LOGERROR("Feature Matching Processor: In ImageValidate without an Image.");
                        _FMPState = FeatureMatchingState_e::FMPState_StreamImages;
                        VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = _FMPState;
                        VPRptr->FeatureMatchProcStatusMsg->NumberOfImagesCaptured = _numberOfFeatureImages;
                        VPRptr->FeatureMatchProcStatusMsg->StatusMessage = "No Image Captured.";
                        featureMatchStatusMsgChanged = true;
                    }
                    //Once we have processed the FeatureMatchProcCtrl Command.. set to null so we
                    //don't end up in a loop reprocessing an old command.
                    VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchingProcCmd = FeatureMatchingProcCmd_e::FMPCmd_NullCmd;
                    break;

                case FeatureMatchingState_e::FMPState_ImageCapturedWait:
                    //Wait for validation from User
                    switch(VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchingProcCmd)
                    {
                        case FeatureMatchingProcCmd_e::FMPCmd_Reset:
                            VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
                            _currImgMsgPtr = nullptr;
                            VPRptr->AddEmptyImageMsgsToQueue();
                            _FMPState = FeatureMatchingState_e::FMPState_Reset;
                            break;

                        case FeatureMatchingProcCmd_e::FMPCmd_SetImageOk:
                        {
                            //Create a local copy of the image
                            _currImgMsgPtr->ImageFrame.copyTo(_imagePrimary);
                            //Save Image
                            string fn = CreateImageFilename(_numberOfFeatureImages + 1);
                            bool storedError = !_JpgFileHandler.CompressAndStoreImage(_currImgMsgPtr->ImageFrame, fn);

                            //Now return our ImagePlusMetadataMessage back to the pool to be used again.
                            VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
                            _currImgMsgPtr = nullptr;
                            VPRptr->AddEmptyImageMsgsToQueue();
                            _FMPState = FeatureMatchingState_e::FMPState_StreamImages;
                            VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = _FMPState;
                            if (storedError)
                            {
                                VPRptr->FeatureMatchProcStatusMsg->StatusMessage = "Error Storing Image.";
                            } else
                            {
                                ++_numberOfFeatureImages;
                                VPRptr->FeatureMatchProcStatusMsg->StatusMessage = "Image Stored.";
                            }
                            VPRptr->FeatureMatchProcStatusMsg->NumberOfImagesCaptured = _numberOfFeatureImages;
                            featureMatchStatusMsgChanged = true;
                        }
                            break;

                        case FeatureMatchingProcCmd_e::FMPCmd_RejectImage:
                        {
                            VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
                            _currImgMsgPtr = nullptr;
                            VPRptr->AddEmptyImageMsgsToQueue();
                            _FMPState = FeatureMatchingState_e::FMPState_StreamImages;
                            VPRptr->FeatureMatchProcStatusMsg->FeatureMatchingState = _FMPState;
                            VPRptr->FeatureMatchProcStatusMsg->NumberOfImagesCaptured = _numberOfFeatureImages;
                            VPRptr->FeatureMatchProcStatusMsg->StatusMessage = "Image Rejected";
                            featureMatchStatusMsgChanged = true;
                        }
                            break;
                    }
                    //Once we have processed the FeatureMatchProcCtrl Command.. set to null so we
                    //don't end up in a loop reprocessing an old command.
                    VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchingProcCmd = FeatureMatchingProcCmd_e::FMPCmd_NullCmd;
                    break;

                case FeatureMatchingState_e::FMPState_FMProcess:

                    if(VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchingProcCmd
                            == FeatureMatchingProcCmd_e::FMPCmd_Reset)
                    {
                        _FMPState = FeatureMatchingState_e::FMPState_Reset;
                    }
                    else
                    {
                        //Process New Images against Feature Set.
                        VPRptr->AddEmptyImageMsgsToQueue();
                        _currImgMsgPtr = VPRptr->GetNextIncomingImagePlusMetadataMessage();
                        if (_currImgMsgPtr != nullptr)
                        {
                            //Do Image Processing Here.
                            ProcessImageFeatures(_currImgMsgPtr);
                            MatchImageFeatures(_imageKeyptsFeaturesQuery, _imageKeyptsFeaturesPrimary);
                            featureMatchStatusMsgChanged = true;

                            //Send a copy of the Image Plus Metadata to the Stream Record Manager.
                            if (VPRptr->StreamRecordEmptyIPMDMsgQueue->GetMessage(tmpMsgPtr))
                            {
                                imgPMetadataMsg1Ptr = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
                                imgPMetadataMsg1Ptr->CopyMessage(_currImgMsgPtr);
                                PostProcessImageToBeSentOut(imgPMetadataMsg1Ptr);
                                //Send the message back to the Stream Manager even if it is bad,
                                //otherwise we will run out of the messages.
                                VPRptr->StreamRecordRxIPMDMsgQueue->AddMessage(imgPMetadataMsg1Ptr);
                            }
                            //Now return our ImagePlusMetadataMessage back to the pool to be used again.
                            VPRptr->CheckinImagePlusMetadataMessageToPool(_currImgMsgPtr);
                            _currImgMsgPtr = nullptr;
                        }
                    }
                    break;

                case FeatureMatchingState_e::FMPState_FMError:
                    if(VPRptr->FeatureMatchProcCtrlMsg->FeatureMatchingProcCmd
                       == FeatureMatchingProcCmd_e::FMPCmd_Reset)
                    {
                        _FMPState = FeatureMatchingState_e::FMPState_Reset;
                    }
                    break;

            }

        }
        catch (std::exception &e)
        {
            LOGERROR("Feature Matching Processor: Exception: " << e.what());
        }

        if(featureMatchStatusMsgChanged)
        {
            VPRptr->FeatureMatchProcStatusMsg->PostMessage();
        }

        return result;
    }




    //Assumes image is in the _currImgMsgPtr
    //and VPRptr->FeatureMatchProcCtrlMsg contains the options for
    //processing the Features.
    bool FeatureMatchingProcessor::ProcessImageFeatures(ImagePlusMetadataMessage *ipmdMsg)
    {
        bool error = true;
        if( ipmdMsg != NULL )
        {
            try
            {
                //Time the process
                _stopwatch1.reset();
                _stopwatch1.start();

                //Select the Type of processing here.
                if(_FMPState == FeatureMatchingState_e::FMPState_FMProcess)
                {
                    _imageKeyptsFeaturesQuery.generateKeypointsAndFeatures(ipmdMsg->ImageFrame);
                }
                else
                {
                    //determine keypoints and features in the image.
                    _imageKeyptsFeaturesPrimary.generateKeypointsAndFeatures(ipmdMsg->ImageFrame);
                }
                _stopwatch1.stop();

                VPRptr->FeatureMatchProcStatusMsg->ProcessTimer_1 = _stopwatch1.getTimeElapsed();

                error = false;
            }
            catch (std::exception &e)
            {
                LOGERROR("ProcessImageFeatures: Exception: " << e.what());
                error = true;
            }
        }

        return error;
    }


    bool FeatureMatchingProcessor::MatchImageFeatures(ImageKeypointsAndFeatures queryInpKptsFeatures,
                                                    ImageKeypointsAndFeatures matchSetInpKptsFeatures)
    {
        bool error = true;
        try
        {
            //Time the process
            _stopwatch2.reset();
            _stopwatch2.start();

            //Modify the image with the feature locations.
            _imageFeatureMatcher.matchFeatures(queryInpKptsFeatures, matchSetInpKptsFeatures);
            _stopwatch2.stop();

            VPRptr->FeatureMatchProcStatusMsg->ProcessTimer_2 = _stopwatch2.getTimeElapsed();
            error = false;
        }
        catch (std::exception &e) {
            LOGERROR("ProcessImageFeatures: Exception: " << e.what());
            error = true;
        }
        return error;
    }


    bool FeatureMatchingProcessor::PostProcessImageToBeSentOut(ImagePlusMetadataMessage *ipmdMsg)
    {
        bool error = true;
        if (ipmdMsg != NULL)
        {
            try
            {
                if(_FMPState == FeatureMatchingState_e::FMPState_FMProcess)
                {
                    if (VPRptr->FeatureMatchProcCtrlMsg->FMImagePostProcessMethod !=
                        FMImagePostProcessMethod_e::FMIPPM_None)
                    {
                        //Modify the image with the feature locations.
                        _imageFeatureMatcher.mapFeaturesAcrossImages(_currImgMsgPtr->ImageFrame,
                                                                     _imageKeyptsFeaturesQuery,
                                                                     _imagePrimary,
                                                                     _imageKeyptsFeaturesPrimary,
                                                                     ipmdMsg->ImageFrame);
                    }
                }
                else
                {
                    if (VPRptr->FeatureMatchProcCtrlMsg->FMImagePostProcessMethod !=
                        FMImagePostProcessMethod_e::FMIPPM_None)
                    {
                        //Modify the image with the feature locations.
                        _imageKeyptsFeaturesPrimary.markImageWithKeypoints(ipmdMsg->ImageFrame);
                    }
                }
                error = false;
            }
            catch (std::exception &e) {
                LOGERROR("ProcessImageFeatures: Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }

}