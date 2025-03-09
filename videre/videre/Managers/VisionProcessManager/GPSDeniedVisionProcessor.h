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

#ifndef VIDERE_DEV_GPSDENIEDVISIONPROCESSOR_H
#define VIDERE_DEV_GPSDENIEDVISIONPROCESSOR_H

#include "VisionProcessResources.h"
#include "VisionProcessorAbstract.h"
#include "TargetDetectorProcessControl.h"


namespace videre
{

    class GPSDeniedVisionProcessor : public VisionProcessorAbstract
    {

     public:
        GPSDeniedVisionProcessor(VisionProcessResources* vpResources);

        ~GPSDeniedVisionProcessor();

        //Reset the Vision Processor.
        //Reset must return all resources obtained during operation...
        //such a image messages, target result messages and the like.
        virtual void Reset();

        //This is the primary call for running the Vision Processor
        //each time through the Manager's loop.
        virtual VPProcessResult_e ExecuteUnitOfWork();

    };

}
#endif //VIDERE_DEV_GPSDENIEDVISIONPROCESSOR_H
