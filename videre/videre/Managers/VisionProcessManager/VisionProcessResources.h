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

#ifndef VIDERE_DEV_VISIONPROCESSRESOURCES_H
#define VIDERE_DEV_VISIONPROCESSRESOURCES_H

#include <iostream>
#include <string>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/features2d/features2d.hpp>
#include <thread>
#include <chrono>
#include <memory>
#include <RabitManager.h>
#include <message_pool.h>

#include "../../Messages/all_manager_message.h"
#include "../../Messages/ImageProcessControlMessage.h"
#include "../../Messages/StreamRecordImagesControlMessage.h"
#include "../../Messages/ImageProcTargetInfoResultsMessage.h"
#include "../../Messages/CameraCalCommandMessage.h"
#include "../../Messages/CameraCalStatusMessage.h"
#include "../../Messages/FeatureMatchProcCtrlMessage.h"
#include "../../Messages/FeatureMatchProcStatusMessage.h"
#include "../../Utils/global_defines.h"
#include "../../Utils/config_data.h"
#include "../../Utils/logger.h"

#include "../StreamRecordManager/vision_record_play.h"
#include "BlobTargetDetectionProcess/BlobTargetDetector.h"
#include "TargetDetectorProcessControl.h"
#include "ImageLoggingControlMessage.h"

namespace videre
{

    //This class contains resources used by the different
    //Vision Processor objects.
    //There should be only one of these objects so that
    //the resources are not duplicated.   Since only one
    //Vision Processor runs at a time with in the Vision Processor
    //Manager... there is no need to thread lock the resources.
    class VisionProcessResources
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        //The reference to the manager is used primarily for
        //setup purposes.
        Rabit::RabitManager* _mgrPtr;

        const int ImagePoolSize = 3;

        //The VisionProcessManager keeps of pool of ImagePlusMetaData
        //messages.  The pool  should be atleast two
        //so that VisionProcessManager can be recording/procesing an image
        //while the ImageCaptureManager is filling a new ImagePlusMetaData message.
        //The VisionProcessManager will typically send empty ImagePlusMetaData messages
        //to the Image Manager and the Image Manager will fill the message with an image
        //plus metadata and send the message back to the VisionProcessManager for processing
        //The VisionProcessManager will return processed messages back to the pool.
        std::unique_ptr<MessagePool> _imageMessagePool_uptr;

        //The VisionProcessManager keeps of pool of ImageProcTargetInfoResultsMessage
        //messages.  After processing an image for targets... Image procesing builds
        //the result information and sends it back to HOPS.  Even if there are no
        //targets found in the image, a message is sent to HOPS to indicate that
        //a image was processed and the ground location of the image.  A pool of the
        //messages is maintained to keep from creating and distroying these messages
        //on every image processed.  The Comms Manager returns the empty messages.
        std::unique_ptr<MessagePool> _targetInfoResultsMsgPool_uptr;

    public:
        //Messages
        std::shared_ptr<ImageProcessControlMessage> ImageProcessControlMsg;

        //Parot back the status of Image Processing
        std::shared_ptr<ImageProcessControlMessage> ImageProcessControlStatusMsg;

        std::shared_ptr<StreamRecordImageControlMesssage> StreamRecordControlMsg;

        std::shared_ptr<CameraCalCommandMessage> CameraCalControlMsg;

        std::shared_ptr<CameraCalStatusMessage> CameraCalStatusMsg;

        std::shared_ptr<FeatureMatchProcCtrlMessage> FeatureMatchProcCtrlMsg;

        std::shared_ptr<FeatureMatchProcStatusMessage> FeatureMatchProcStatusMsg;

        std::shared_ptr<ImageLoggingControlMessage> LoggingControlMsg;

        //Queues
        //This Queue is to recieve Image Plus Metadata messages... typically
        //from the Image Manager.
        std::shared_ptr<RabitMsgPtrSPSCQueue> RxImagePlusMetadataQueue;

        //This Queue is to send Emtpty Image Plus Metadata messages... typically
        //to the Image Manager.
        std::shared_ptr<RabitMsgPtrSPSCQueue> EmptyImagePlusMetadataQueue;

        //This Queue is to recieve Image Plus Metadata messages... typically
        //from the Vision Process Manage.
        std::shared_ptr<RabitMsgPtrSPSCQueue> StreamRecordRxIPMDMsgQueue;

        //This Queue is to send Emtpty Image Plus Metadata messages...
        //to the Vision Process Manager.
        std::shared_ptr<RabitMsgPtrSPSCQueue> StreamRecordEmptyIPMDMsgQueue;

        //This Queue is to recieve Image Plus Metadata messages... typically
        //from the Image Manager.
        std::shared_ptr<RabitMsgPtrSPSCQueue> TgtMsgEmptyQueue;

        //This Queue is to send Emtpty Image Plus Metadata messages... typically
        //to the Image Manager.
        std::shared_ptr<RabitMsgPtrSPSCQueue> TgtMsgsToBeSentQueue;

        bool ImageProcessControlMsgChanged = false;
        bool ImageProcessControlStatusMsgChanged = false;

        bool PostProcessImages = false;

        std::string CameraCalDataDirectory;

        std::string CameraCalDataFilename;

        bool CalibrationIsSetup = false;

        ImageProcLibsNS::CameraCalibrationData CameraCalData;


    public:
        VisionProcessResources(Rabit::RabitManager* mgrPtr,
                               std::shared_ptr<ConfigData> config);

        ~VisionProcessResources();

        void Initialize();

        std::shared_ptr<ConfigData> GetConfig()
        {
            return _config_sptr;
        }

        Rabit::RabitManager* GetMgrPtr()
        {
            return _mgrPtr;
        }

        bool ReadCameraCalDataFromFile(const std::string &directory, const std::string &filename);

        ImagePlusMetadataMessage* CheckoutImagePlusMetadataMessageFromPool();

        void CheckinImagePlusMetadataMessageToPool(ImagePlusMetadataMessage* msg);

        int GetNumberOfImagePlusMetadataMessagesInPool();

        ImageProcTargetInfoResultsMessage* CheckoutImageProcTargetInfoResultsMessageFromPool();

        void CheckinImageProcTargetInfoResultsMessageToPool(ImageProcTargetInfoResultsMessage* msg);

        int GetNumberOfImageProcTargetInfoResultsMessagesInPool();

        //Add Empty ImagePlusDataMessages from the IMage Message Pool
        //to the EmptyImagePlusMetadataQueue.
        //Empty messages are only added if some level of the Stream
        //Manager is enabled.  This keeps the Image Manager from sending
        //this manager images when they are not being used.
        //Controlling the rate/timing of adding Empty messages to the
        //Queue is one way to throttle the rate that this manager is processing
        //images.
        void AddEmptyImageMsgsToQueue();

        //Get empty target messages from Queue and put them back in the pool.
        void CheckForEmptyTargetMessages();

        //Get the next incoming ImagePlusMetadataMessage from the Image Capture
        //Manager.  Returns pointer to the image or null.
        //ImagePlusMetadataMessages must be returned to the ImagePlusMetadataMessage
        //Pool.
        ImagePlusMetadataMessage* GetNextIncomingImagePlusMetadataMessage();

        //Fetch... Update the ImageProcessControlMsg.
        //This sets the ImageProcessControlMsgChanged flag.
        //If the message changed... it also sets all the
        //ImageProcessControlStatusMsg parameters base on the
        //values in ImageProcessControlMsg.
        bool FetchImageProcessControlMsgCheckChanged();

        bool SetImageProcessControlStatusMsgDefaults(bool setMsgChangedFlag);

        void PostIfChangedImageProcessControlStatusMsg()
        {
            if(ImageProcessControlStatusMsgChanged)
            {
                ImageProcessControlStatusMsg->PostMessage();
                ImageProcessControlStatusMsgChanged = false;
            }
        }

        void SetTargetProcesModeInImageProcessControlStatusMsg(TargetProcessingMode_e tgtPMode)
        {
            if( ImageProcessControlStatusMsg->TargetProcessingMode != tgtPMode)
            {
                ImageProcessControlStatusMsg->TargetProcessingMode = tgtPMode;
                ImageProcessControlStatusMsgChanged = true;
            }
        }

        void SetGPUProcessingEnabledInImageProcessControlStatusMsg(bool gpuEnabled)
        {
            if( ImageProcessControlStatusMsg->GPUProcessingEnabled != gpuEnabled)
            {
                ImageProcessControlStatusMsg->GPUProcessingEnabled = gpuEnabled;
                ImageProcessControlStatusMsgChanged = true;
            }
        }

    };

}
#endif //VIDERE_DEV_VISIONPROCESSRESOURCES_H
