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

#ifndef VIDERE_DEV_CAMERACALIBRATIONPROCESSCONTROL_H
#define VIDERE_DEV_CAMERACALIBRATIONPROCESSCONTROL_H

#include <string>
#include <RabitManager.h>
#include "logger.h"
#include "config_data.h"

using namespace videre;

namespace CameraCalibrationNS
{
    //The Camera Calibration Process Control
    //Helps guide the process of camera calibration and storing the calibaration
    //results for later use.
    class CameraCalibrationProcessControl
    {
    private:

        std::shared_ptr<ConfigData> _config_sptr;

        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        //The reference to the manager is used primarily for
        //setup purposes.
        Rabit::RabitManager* _mgrPtr;

    public:
        CameraCalibrationProcessControl(Rabit::RabitManager* mgrPtr,
        std::shared_ptr<ConfigData> config);

        ~CameraCalibrationProcessControl();


    };

}

#endif //VIDERE_DEV_CAMERACALIBRATIONPROCESSCONTROL_H
