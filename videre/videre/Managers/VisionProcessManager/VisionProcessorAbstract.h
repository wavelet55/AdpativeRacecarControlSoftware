/* ****************************************************************
 * Vision Processor Abstract Class
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

#ifndef VIDERE_DEV_VISIONPROCESSORABSTRACT_H
#define VIDERE_DEV_VISIONPROCESSORABSTRACT_H

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
#include "VisionProcessResources.h"
#include "../../Messages/all_manager_message.h"
#include "../../Messages/ImageProcessControlMessage.h"
#include "../../Messages/StreamRecordImagesControlMessage.h"
#include "../../Messages/ImageProcTargetInfoResultsMessage.h"
#include "../../Utils/global_defines.h"
#include "../../Utils/config_data.h"
#include "../../Utils/logger.h"

#include "../StreamRecordManager/vision_record_play.h"
#include "BlobTargetDetectionProcess/BlobTargetDetector.h"
#include "TargetDetectorProcessControl.h"


namespace videre
{
    enum VPProcessResult_e
    {
        VPPR_none = 0,      //No change or no result

        VPPR_error          //There is an error... Reset the Processor before running again.
    };

    //This is the abstract class that all vision processors must inherit from.
    //To allow various types of Vision or image processors to work with the
    //system and run under the Vision Process manager, a generic processor class
    //is established here.  This class provides all the top-level structure each
    //Vision Processor can inherit from.  In the current design, only one
    //Vision Processor should be run at any time.  Different Vision Processors
    //with different Image Processing algorithms can be run by choosing a specific
    //Vision Processor.  The primary reason for establishing Vision Processors is that
    //some vision processors need to be state-based... carring out different portions
    //of there routine over time.  Setting up the Vision Processor model allows
    //each Vision Processor to support what ever level of state-based processing
    //it needs.
    //The ExecuteUnitOfWork() method is the primary call to the Vision Processor
    //each time around the Vision Process manager.  This method must be established
    //for each Vision Processor.
    //The ResetProcessor() must be provided.  This method can be called at anytime.
    //The Vision Processor must clean up and restore any resources, such as image
    //messages pulled from the image message pool.
    //
    class VisionProcessorAbstract
    {
    protected:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        VisionProcessResources*  VPRptr;


    public:
        VisionProcessorAbstract(VisionProcessResources* vpResources)
        {
            VPRptr = vpResources;
            log4cpp_ = log4cxx::Logger::getLogger("aobj");
            log4cpp_->setAdditivity(false);
        }


        //Intialize any resources at the start of operation.
        virtual void Intialize() {}

        //Reset the Vision Processor.
        //Reset must return all resources obtained during operation...
        //such a image messages, target result messages and the like.
        virtual void Reset() = 0;

        //This is the primary call for running the Vision Processor
        //each time through the Manager's loop.
        virtual VPProcessResult_e ExecuteUnitOfWork() = 0;

    protected:
        //Common helper routines for the Vission Processors.


        //ImagePlusMetadataMessage GetNewImagePlusMetadata();
    };

}


#endif //VIDERE_DEV_VISIONPROCESSORABSTRACT_H
