/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * Randy Direen, PhD
 *
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Aug, 2018
 *
 * Head Orientation Vehicle Control Process.
 *******************************************************************/


#ifndef VIDERE_DEV_HEADORIENTATIONVEHICLECONTROLPROCESS_H
#define VIDERE_DEV_HEADORIENTATIONVEHICLECONTROLPROCESS_H

#include <memory>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <RabitManager.h>
#include <RabitStopWatch.h>
#include <RabitMessageQueue.h>
#include "global_defines.h"
#include "all_manager_message.h"
#include "../../Utils/config_data.h"
#include "../../Utils/logger.h"
#include "../../Utils/timing.h"
#include "DataRecorder.h"
#include "DataRecorderStdHeader.h"
#include "VidereSystemControlDataRecord.h"
#include "ImageLoggingControlMessage.h"
#include "QuaternionMessage.h"
#include "Quaternion.h"
#include <armadillo>
#include "SystemControlResources.h"
#include "ButterworthIIRLPF.h"

using namespace Rabit;

namespace videre
{

    class HeadOrientationVehicleControlProcess
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        //The reference to the manager is used primarily for
        //setup purposes.
        Rabit::RabitManager* _mgrPtr;

        SystemControlResources *_SCRPtr;

        Rabit::RabitStopWatch _stopwatch;

        double _throttleCtrlPercent = 0.0;
        double _brakeCtrlPercent = 0.0;
        //brakeThrottleIntegral... throttle is positive values
        //Brake is negagive values;  range: [-100, 100]
        double _throttleBrakeIntegral = 0;
        double _steeringTorqueControlPercent = 0.0;

        MathLibsNS::ButterworthIIRLPF _headLRLPF;


    public:
        HeadOrientationVehicleControlProcess(Rabit::RabitManager* mgrPtr,
                                             std::shared_ptr<ConfigData> config,
                                             SystemControlResources *_SCRPtr);

        ~HeadOrientationVehicleControlProcess() {}

        bool Initialize();

        void SetupHeadLeftRighLPF();

        void Shutdown();

        //Reset should be called when leaving and before
        //entering the HeadOrientationVehicleControlProcess state of operation.
        void Reset();

        SystemStateResponse_t  ExecuteUnitOfWork();

        double SteeringHeadAngleCtrl(double dt, const XYZCoord_t &eulerAnglesDegrees);

        double SteeringAngleFeedbackLoop(double headAngleDegrees, double dt);

        void ThrottleBrakeSipnPuffControl();

        void ThrottleBrakeHeadTiltSipnPuffControl(double dt, const XYZCoord_t &eulerAnglesDegrees);

        void SetBrakeAcutator();

        void SetThrottleAcutator();

        void SetSteeringControlActuator();

    };

}
#endif //VIDERE_DEV_HEADORIENTATIONVEHICLECONTROLPROCESS_H
