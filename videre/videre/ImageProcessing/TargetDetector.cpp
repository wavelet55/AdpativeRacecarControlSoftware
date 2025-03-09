/* ****************************************************************
 * Blob Target Detector
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Nov. 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
   *******************************************************************/

#include "TargetDetector.h"

namespace TargetDetectorNS
{


    TargetDetector::TargetDetector(Rabit::RabitManager* mgrPtr,
                                   std::shared_ptr<ConfigData> config)
    {
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        _targetType1ParamsMsg = std::make_shared<TargetParametersMessage>();
        mgrPtr->AddPublishSubscribeMessage("TargetType1ParamsMsg", _targetType1ParamsMsg);

        _targetType2ParamsMsg = std::make_shared<TargetParametersMessage>();
        mgrPtr->AddPublishSubscribeMessage("TargetType2ParamsMsg", _targetType2ParamsMsg);

        _targetType3ParamsMsg = std::make_shared<TargetParametersMessage>();
        mgrPtr->AddPublishSubscribeMessage("TargetType3ParamsMsg", _targetType3ParamsMsg);

        _targetType4ParamsMsg = std::make_shared<TargetParametersMessage>();
        mgrPtr->AddPublishSubscribeMessage("TargetType4ParamsMsg", _targetType4ParamsMsg);

        _imageProcessingParamsMsg = std::make_shared<FeatureMatchProcCtrlMessage>();
        mgrPtr->AddPublishSubscribeMessage("FeatureMatchProcCtrlMessage", _imageProcessingParamsMsg);
    }


}

