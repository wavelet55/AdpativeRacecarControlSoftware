/* ****************************************************************
 * Camera Calibration Process Control
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

#include "CameraCalibrationProcessControl.h"

namespace CameraCalibrationNS
{

    CameraCalibrationProcessControl::CameraCalibrationProcessControl(Rabit::RabitManager* mgrPtr,
                                                               std::shared_ptr<ConfigData> config)
    {
        _mgrPtr = mgrPtr;
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
        LOGINFO("TargetDetectorProcessControl Created.");

    }


    CameraCalibrationProcessControl::~CameraCalibrationProcessControl()
    {

    }

}