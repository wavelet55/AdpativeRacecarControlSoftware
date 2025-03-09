/* ****************************************************************
 * Vision Processor Resources
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


#include "VisionProcessResources.h"
#include "FileUtils.h"
#include "CameraCalibration/CameraCalReaderWriter.h"

namespace  videre
{


    VisionProcessResources::VisionProcessResources(Rabit::RabitManager* mgrPtr,
                                                std::shared_ptr<ConfigData> config)
    {
        _mgrPtr = mgrPtr;
        _config_sptr = config;
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        //Messages
        ImageProcessControlMsg = std::make_shared<ImageProcessControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageProcessControlMessage", ImageProcessControlMsg);

        ImageProcessControlStatusMsg = std::make_shared<ImageProcessControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageProcessControlStatusMessage", ImageProcessControlStatusMsg);

        StreamRecordControlMsg = std::make_shared<StreamRecordImageControlMesssage>();
        _mgrPtr->AddPublishSubscribeMessage("StreamRecordImageControlMesssage", StreamRecordControlMsg);

        CameraCalControlMsg = std::make_shared<CameraCalCommandMessage>();
        _mgrPtr->AddPublishSubscribeMessage("CameraCalCommandMessage", CameraCalControlMsg);

        CameraCalStatusMsg = std::make_shared<CameraCalStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage("CameraCalStatusMessage", CameraCalStatusMsg);

        FeatureMatchProcCtrlMsg = std::make_shared<FeatureMatchProcCtrlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("FeatureMatchProcCtrlMessage", FeatureMatchProcCtrlMsg);

        FeatureMatchProcStatusMsg = std::make_shared<FeatureMatchProcStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage("FeatureMatchProcStatusMessage", FeatureMatchProcStatusMsg);

        LoggingControlMsg = std::make_shared<ImageLoggingControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageLoggingControlMessage", LoggingControlMsg);

        //Queues
        RxImagePlusMetadataQueue = std::make_shared<RabitMsgPtrSPSCQueue>(
                ImagePoolSize,
                "VissionProcMgrRxImagePlusMetadataMsgQueue");
        _mgrPtr->AddManagerMessageQueue(RxImagePlusMetadataQueue->GetMessageQueueName(),
                                     RxImagePlusMetadataQueue);

        //Set event that will wakeup the loop when we receive a new Image.
        //this->WakeUpManagerOnEnqueue(_RxImagePlusMetadataQueue_sptr);

        EmptyImagePlusMetadataQueue = std::make_shared<RabitMsgPtrSPSCQueue>(
                ImagePoolSize,
                "VissionProcMgrEmptyImagePlusMetadataMsgQueue");
        _mgrPtr->AddManagerMessageQueue(EmptyImagePlusMetadataQueue->GetMessageQueueName(),
                                        EmptyImagePlusMetadataQueue);


        TgtMsgsToBeSentQueue = std::make_shared<RabitMsgPtrSPSCQueue>(
                ImagePoolSize + 1,
                "ImageProcTgtMsgQueue");
        _mgrPtr->AddManagerMessageQueue(TgtMsgsToBeSentQueue->GetMessageQueueName(),
                                        TgtMsgsToBeSentQueue);

        TgtMsgEmptyQueue = std::make_shared<RabitMsgPtrSPSCQueue>(
                ImagePoolSize + 1,
                "ImageProcTgtMsgEmptyQueue");
        _mgrPtr->AddManagerMessageQueue(TgtMsgEmptyQueue->GetMessageQueueName(),
                                        TgtMsgEmptyQueue);

        //Setup the Image Plus Metadata Message Pool.
        //We temporarity need to create a ImagePlusMetaData Message required to setup the pool.
        //The message is only needed for the setup process and can then be discarded.
        ImagePlusMetadataMessage ipmMsg;
        _imageMessagePool_uptr = unique_ptr<MessagePool>(new MessagePool(&ipmMsg, ImagePoolSize));

        //Setup the Image Plus Metadata Message Pool.
        //We temporarity need to create a ImagePlusMetaData Message required to setup the pool.
        //The message is only needed for the setup process and can then be discarded.
        ImageProcTargetInfoResultsMessage iptgtMsg;
        _targetInfoResultsMsgPool_uptr = unique_ptr<MessagePool>(new MessagePool(&iptgtMsg, ImagePoolSize + 1));

        CameraCalDataDirectory = _config_sptr->GetConfigStringValue("CameraCal.CameraCalDataDirectory", "CameraCalData");
        //Create a CameraCalDirectory if it does not exist.
        VidereFileUtils::CreateDirectory(CameraCalDataDirectory);

        CameraCalDataFilename = _config_sptr->GetConfigStringValue("CameraCal.DefaultCameraCalFilename", "CameraDefaultCalData");
        CameraCalDataFilename = VidereFileUtils::AddOrReplaceFileExtention(CameraCalDataFilename, CAMERA_CAL_FILE_EXT);

    }

    VisionProcessResources::~VisionProcessResources()
    {

    }

    void VisionProcessResources::Initialize()
    {

        try
        {
            CameraCalDataDirectory = _config_sptr->GetConfigStringValue("CameraCal.CameraCalDataDirectory", "CameraCalData");
            CameraCalDataFilename = _config_sptr->GetConfigStringValue("CameraCal.DefaultCameraCalFilename", "CameraDefaultCalData");
            CameraCalDataFilename = VidereFileUtils::AddOrReplaceFileExtention(CameraCalDataFilename, CAMERA_CAL_FILE_EXT);
            CalibrationIsSetup = false;
            ReadCameraCalDataFromFile(CameraCalDataDirectory, CameraCalDataFilename);

            PostProcessImages = _config_sptr->GetConfigBoolValue("flags.PostProcessImages", "true");

            //Connect to Record Stream Message Queues
            StreamRecordRxIPMDMsgQueue = _mgrPtr->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>(
                    "StreamRecordRxIPMDMsgQueue");
            StreamRecordEmptyIPMDMsgQueue = _mgrPtr->GetManagerMessageQueue<std::shared_ptr<RabitMsgPtrSPSCQueue>>(
                    "StreamRecordEmptyIPMDMsgQueue");
        }
        catch (MessageNotRegisteredException &e)
        {
            LOGWARN("VisionProcessManager: The StreamRecordRxIPMDMsgQueue message queue does not exist.");
            cout << "VisionProcessManager: The StreamRecordRxIPMDMsgQueue message queue does not exist." << endl;
        }
    }

    bool VisionProcessResources::ReadCameraCalDataFromFile(const std::string &directory, const std::string &filename)
    {
        bool error = false;
        try
        {
            boost::filesystem::path fullfilename(directory);
            fullfilename /= VidereFileUtils::AddOrReplaceFileExtention(filename, CAMERA_CAL_FILE_EXT);
            CameraCalibrationNS::ReadCameraCalibrationFromIniFile(fullfilename.c_str(), CameraCalData);
            LOGINFO("Camera Cal Data Read from: " << filename);
            CalibrationIsSetup = true;
        }
        catch(exception e)
        {
            error = true;
            LOGWARN("Camera Cal Data Read Error, file: " << filename << " Ex:" << e.what());
            CameraCalData.SetDefaults();
            CalibrationIsSetup = false;
        }
        return error;
    }


    ImagePlusMetadataMessage* VisionProcessResources::CheckoutImagePlusMetadataMessageFromPool()
    {
        ImagePlusMetadataMessage* ipmdMsg = nullptr;
        if(_imageMessagePool_uptr->GetNumberOfFreeMessages() > 0)
        {
            RabitMessage *emptyMsgPtr = _imageMessagePool_uptr->CheckOutMessage();
            if (emptyMsgPtr != nullptr)
            {
                ipmdMsg = static_cast<ImagePlusMetadataMessage *>(emptyMsgPtr);
            }
        }
        return ipmdMsg;
    }

    void VisionProcessResources::CheckinImagePlusMetadataMessageToPool(ImagePlusMetadataMessage* msg)
    {
        _imageMessagePool_uptr->CheckInMessage(msg);
    }

    int VisionProcessResources::GetNumberOfImagePlusMetadataMessagesInPool()
    {
        return _imageMessagePool_uptr->GetNumberOfFreeMessages();
    }

    ImageProcTargetInfoResultsMessage* VisionProcessResources::CheckoutImageProcTargetInfoResultsMessageFromPool()
    {
        ImageProcTargetInfoResultsMessage* tgtMsg = nullptr;
        if(_targetInfoResultsMsgPool_uptr->GetNumberOfFreeMessages() > 0)
        {
            RabitMessage *emptyMsgPtr = _targetInfoResultsMsgPool_uptr->CheckOutMessage();
            if (emptyMsgPtr != nullptr)
            {
                tgtMsg = static_cast<ImageProcTargetInfoResultsMessage *>(emptyMsgPtr);
            }
        }
        return tgtMsg;
    }

    void VisionProcessResources::CheckinImageProcTargetInfoResultsMessageToPool(ImageProcTargetInfoResultsMessage* msg)
    {
        _targetInfoResultsMsgPool_uptr->CheckInMessage(msg);
    }

    int VisionProcessResources::GetNumberOfImageProcTargetInfoResultsMessagesInPool()
    {
        return _targetInfoResultsMsgPool_uptr->GetNumberOfFreeMessages();
    }


    void VisionProcessResources::AddEmptyImageMsgsToQueue()
    {
        if((ImageProcessControlMsg->TargetImageProcessingEnabled
            || ImageProcessControlMsg->GPSDeniedProcessingEnabled)
           &&( _imageMessagePool_uptr->GetNumberOfFreeMessages() > 0))
        {
            //Add a empty message to the queue
            //The message queue was created large enough to handle the size of the
            //Image Message Pool, so we don't need to check to see if the queue is full.
            RabitMessage* emptyMsgPtr = _imageMessagePool_uptr->CheckOutMessage();
            if(emptyMsgPtr != nullptr)
            {
                if( !EmptyImagePlusMetadataQueue->AddMessage(emptyMsgPtr) )
                {
                    //This should not happen.
                    LOGWARN("VisionProcessManager: EmptyImagePlusMetadataQueue is Full!")
                    _imageMessagePool_uptr->CheckInMessage(emptyMsgPtr);
                }
            }
        }
    }

    //Get any empty Image Messages from the Empty Image Msg Queue and
    //add them back to the Pool.
    void VisionProcessResources::CheckForEmptyTargetMessages()
    {
        RabitMessage *emptyImgMsg;
        while( TgtMsgEmptyQueue->GetMessage(emptyImgMsg))
        {
            _targetInfoResultsMsgPool_uptr->CheckInMessage(emptyImgMsg);
        }
    }

    //Get the next incoming ImagePlusMetadataMessage from the Image Capture
    //Manager.  Returns pointer to the image or null.
    //ImagePlusMetadataMessages must be returned to the ImagePlusMetadataMessage
    //Pool.
    ImagePlusMetadataMessage* VisionProcessResources::GetNextIncomingImagePlusMetadataMessage()
    {
        Rabit::RabitMessage *tmpMsgPtr;
        ImagePlusMetadataMessage* ipmdMsg = nullptr;
        if(RxImagePlusMetadataQueue->GetMessage(tmpMsgPtr))
        {
            try
            {
                ipmdMsg = static_cast<ImagePlusMetadataMessage *>(tmpMsgPtr);
                if(ipmdMsg->ImageNumber == 0
                        || ipmdMsg->ImageFrame.data == NULL
                        || ipmdMsg->ImageFrame.rows == 0
                        || ipmdMsg->ImageFrame.cols == 0 )
                {
                    //Invalid image... return message to pool
                    CheckinImagePlusMetadataMessageToPool(ipmdMsg);
                    ipmdMsg = nullptr;
                }
            }
            catch(std::exception &e)
            {
                LOGERROR("VisionProcessResources.GetNextIncomingImagePlusMetadataMessage: Exception: " << e.what());
                ipmdMsg = nullptr;
            }
        }
        return ipmdMsg;
    }

    bool VisionProcessResources::FetchImageProcessControlMsgCheckChanged()
    {
        ImageProcessControlStatusMsgChanged = false;
        ImageProcessControlMsgChanged = ImageProcessControlMsg->FetchMessage();
        if(ImageProcessControlMsgChanged )
        {
            SetImageProcessControlStatusMsgDefaults(true);
        }
        return ImageProcessControlMsgChanged;
    }

    bool VisionProcessResources::SetImageProcessControlStatusMsgDefaults(bool setMsgChangedFlag)
    {
        ImageProcessControlStatusMsg->Clear();
        ImageProcessControlStatusMsg->VisionProcessingMode = ImageProcessControlMsg->VisionProcessingMode;
        ImageProcessControlStatusMsg->TargetImageProcessingEnabled = ImageProcessControlMsg->TargetImageProcessingEnabled;
        ImageProcessControlStatusMsg->GPSDeniedProcessingEnabled = ImageProcessControlMsg->GPSDeniedProcessingEnabled;
        ImageProcessControlStatusMsgChanged = setMsgChangedFlag;
        return true;
    }

}