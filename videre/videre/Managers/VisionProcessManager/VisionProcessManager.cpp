/* ****************************************************************
 * Stream Record Manager
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: May 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#include <image_plus_metadata_message.h>
#include "VisionProcessManager.h"
#include "NullVisionProcessor.h"
#include "StdTargetDectectorVisionProcessor.h"
#include "GPSDeniedVisionProcessor.h"
#include "CameraCalibrationProcessor.h"
#include "FileUtils.h"

using namespace std;
using namespace cv;
using namespace Rabit;


namespace videre
{

    VisionProcessManager::VisionProcessManager(std::string name, std::shared_ptr<ConfigData> config)
    : VisionProcessManagerWSRMgr(name),
              _VPResources(this, config),
              _NullVisionProcessor(&_VPResources),
              _StdTargetDectectorVisionProcessor(&_VPResources),
              _GPSDeniedVisionProcessor(&_VPResources),
              _CameraCalibrationProcessor(&_VPResources),
              _FeatureMatchingProcessor(&_VPResources),
              _HeadTrackingProcessor(&_VPResources),
              _HeadOrientationCalProcessor(&_VPResources)
    {
        this->SetWakeupTimeDelayMSec(250);
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        //Messages

         //Create a List of Vision Processors.
        //Whenever new vision processors are created... they must be added to the list.
        //The list is in order of and indexed by the VisionProcessingMode_e
        _VisionProcessorList[0] = &_NullVisionProcessor;
        _VisionProcessorList[1] = &_StdTargetDectectorVisionProcessor;
        _VisionProcessorList[2] = &_GPSDeniedVisionProcessor;
        _VisionProcessorList[3] = &_CameraCalibrationProcessor;
        _VisionProcessorList[4] = &_FeatureMatchingProcessor;
        _VisionProcessorList[5] = &_HeadTrackingProcessor;
        _VisionProcessorList[6] = &_HeadOrientationCalProcessor;

        _currentVisionProcessor = &_NullVisionProcessor;
        _currentVisionProcessingMode = VisionProcessingMode_e::VisionProcMode_None;
    }

    void VisionProcessManager::Initialize()
    {

        LOGINFO("VisionProcessManager: Initialization Started")
        try
        {
            _VPResources.Initialize();

            //Ensure we have a directory setup for the Calibration data to be stored.
            VidereFileUtils::CreateDirectory(_config_sptr->GetConfigStringValue("CameraCalDataDirectory", "CameraCalData"));

            //This parameter should not change during normal operation.
            _postProcessImages = _config_sptr->GetConfigBoolValue("flags.PostProcessImages", "true");
            _VPResources.PostProcessImages = _postProcessImages;

            for(int i = 0; i < NumberOFVisionProcessors; i++)
            {
                _VisionProcessorList[i]->Intialize();
                _VisionProcessorList[i]->Reset();
            }

            _VPResources.ImageProcessControlMsg->Register_SomethingPublished(boost::bind(&VisionProcessManager::WakeUpManagerEH, this));

            //Register a wake-up call when an image is placed in the RxImagePlusMetadataQueue;
            this->WakeUpManagerOnEnqueue(_VPResources.RxImagePlusMetadataQueue);

            this->WakeUpManagerOnMessagePost(_VPResources.ImageProcessControlMsg);
            this->WakeUpManagerOnMessagePost(_VPResources.CameraCalControlMsg);

        }
        catch (MessageNotRegisteredException &e)
        {
            LOGWARN("VisionProcessManager: Initialization exception:" << e.what());
            cout << "VisionProcessManager: Initialization exception:" << e.what() << endl;
        }


#ifdef CUDA
        std::cout << "VisionProcessManager: NVidia CUDA Enabled" << std::endl;
        LOGINFO("VisionProcessManager: NVidia CUDA Enabled")
#else
        std::cout << "VisionProcessManager: NVidia CUDA Disabled" << std::endl;
        LOGINFO("VisionProcessManager: NVidia CUDA Disabled")
#endif
        //_blobTargetDetector.Initialize();

        LOGINFO("VisionProcessManager: Initialization Complete")
        std::cout << "VisionProcessManager: Initialization Complete" << std::endl;
    }

    void VisionProcessManager::ExecuteUnitOfWork()
    {
        bool ipcMsgChanged = _VPResources.FetchImageProcessControlMsgCheckChanged();

        try
        {
            if( _VPResources.ImageProcessControlMsg->VisionProcessingMode != _currentVisionProcessingMode)
            {
                LOGINFO("Vision Processor change from: " << _currentVisionProcessingMode
                        << " To: " << _VPResources.ImageProcessControlMsg->VisionProcessingMode);
                cout <<"Vision Processor change from: " << _currentVisionProcessingMode
                     << " To: " << _VPResources.ImageProcessControlMsg->VisionProcessingMode << endl;

                //Reset the current Vision Processor so that it can return all
                //resources being used, before going on to the new vision processor.
                _currentVisionProcessor->Reset();
                int vpm = (int) _VPResources.ImageProcessControlMsg->VisionProcessingMode;
                vpm = vpm < 0 ? 0 : vpm >= NumberOFVisionProcessors ? NumberOFVisionProcessors - 1 : vpm;

                _currentVisionProcessingMode = _VPResources.ImageProcessControlMsg->VisionProcessingMode;
                _currentVisionProcessor = _VisionProcessorList[vpm];
                _currentVisionProcessor->Reset();  //Ensure it starts from reset.

                _VPResources.ImageProcessControlMsg->VisionProcessingMode = _currentVisionProcessingMode;
                _VPResources.SetImageProcessControlStatusMsgDefaults(true);
            }
            if( _currentVisionProcessor != nullptr)
            {
                _currentVisionProcessor->ExecuteUnitOfWork();
                _VPResources.PostIfChangedImageProcessControlStatusMsg();
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("VisionProcessManager: Exception: " << e.what());
        }
   }


}
