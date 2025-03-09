/* ****************************************************************
 * Athr(s): Randy Direen, PhD
 * Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Aug, 2018
 *
 * Videre System Control Manager
 * This is the top-level system control of the racecar
 *******************************************************************/

#ifndef VIDERE_DEV_VIDERESYSTEMCONTROLMANAGER_H
#define VIDERE_DEV_VIDERESYSTEMCONTROLMANAGER_H

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <RabitManager.h>
#include <RabitStopWatch.h>
#include <RabitMessageQueue.h>
#include <ManagerStatusMessage.h>
#include <ManagerControlMessage.h>
#include <ManagerStats.h>
#include <ManagerStatusMessage.h>
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
#include "HeadOrientationVehicleControlProcess.h"
#include "ResetOrientationStateMessage.h"

// Manually include this file that has been autogenerated
#include "VidereSystemControlManagerWSRMgr.h"

using namespace Rabit;

namespace videre
{

    class VidereSystemControlManager : public VidereSystemControlManagerWSRMgr
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        SystemControlResources SCR;

        HeadOrientationVehicleControlProcess _hovcProcess;

        DataRecorder _dataRecorder;

        bool EnableVidereSystemControlfLogging = true;

        VidereSystemStates_e _systemState = VidereSystemStates_e::VSS_Init;

        VidereSystemStates_e _startupDefaultSystemState = VidereSystemStates_e::VSS_HeadOrientationControl;

        uint32_t _ctrlLoopCount = 0;

        FILE* _gpio1PullupOutput = nullptr;
        FILE* _gpio2DriverEnalbeInput = nullptr;
        FILE* _gpio4AuxEnalbeInput = nullptr;

        int _gpio2DriverEnalbeDebounceCntr = 0;
        int _gpio4AuxEnalbeDebounceCntr = 0;

        bool TestIgnoreDriverSw = false;

        bool _enableLoggingDuringHeadControl = false;


    public:
        VidereSystemControlManager(std::string name, std::shared_ptr<ConfigData> config);

        virtual void Initialize();

        virtual void ExecuteUnitOfWork() final;

        //The Startup method is called once when the manager thread is first
        //started.  It can be used for any necessary initialization processess
        //that have to be done after the manager's constructor.
        virtual void Startup() final;

        virtual void Shutdown() final;

        //This routine should be called when switching states to ensure the
        //Brake, Throttle, and Steering actuators are disable and not left in
        //some running-actuated state.
        void ResetVehicleActuators();

        void ReadVehicleInputSwitches();

        SystemStateResponse_t InitState();

        //Returns true if the system needs to switch to safety driver control.
        bool CheckForSwitchToSafetyDriverControl();

        //Must be called before switching to a new state.
        void ResetSubStates();

        SystemStateResponse_t RemoteControlState();

        SystemStateResponse_t ExternalMonitorControlState();

        SystemStateResponse_t ManualDriverControlState();

    };

}
#endif //VIDERE_DEV_VIDERESYSTEMCONTROLMANAGER_H
