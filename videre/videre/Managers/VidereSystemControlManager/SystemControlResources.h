/* ****************************************************************
 * System Control Resources
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug. 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_SYSTEMCONTROLRESOURCES_H
#define VIDERE_DEV_SYSTEMCONTROLRESOURCES_H

#include <chrono>
#include <memory>
#include <RabitManager.h>
#include <message_pool.h>
#include "../../Messages/all_manager_message.h"
#include "../../Utils/global_defines.h"
#include "../../Utils/config_data.h"
#include "../../Utils/logger.h"
#include "ImageLoggingControlMessage.h"
#include "SteeringTorqueCtrlMessage.h"
#include "DceEPASteeringStatusMessage.h"
#include "LinearActuatorPositionCtrlMessage.h"
#include "VidereSystemControlMessage.h"
#include "VehicleControlParametersMessage.h"
#include "QuaternionMessage.h"
#include "GPSFixMessage.h"
#include "Quaternion.h"
#include "SipnPuffMessage.h"
#include <armadillo>
#include "RemoteControlInputMessage.h"
#include "SipnPuffControlMessage.h"
#include "VehicleSwitchInputMessage.h"
#include "VidereSystemControlDataRecord.h"
#include "DataRecorderStdHeader.h"
#include "ImageProcessControlMessage.h"
#include "ImageCaptureControlMessage.h"
#include "HeadOrientationControlMessage.h"
#include "ResetOrientationStateMessage.h"
#include "StreamRecordImagesControlMessage.h"


namespace videre
{


    enum SystemStateResponse_e
    {
        SSR_Ok,
        SSR_ChangeStateRequest,
        SSR_Error
    };

    struct SystemStateResponse_t
    {
        SystemStateResponse_e Response;
        VidereSystemStates_e NextDesiredState;

        SystemStateResponse_t()
        {
            Clear();
        }

        void Clear()
        {
            Response = SystemStateResponse_e::SSR_Ok;
            NextDesiredState = VidereSystemStates_e::VSS_ManualDriverControl;
        }
    };

    enum ManualDriverControlState_e
    {
        MDCS_Reset,
        MDCS_DriverControlSWDisabled,
        MDCS_DriverControlSWEnabled,
    };

    struct ValueAtTime_t
    {
        double value;
        double tsec;

        void clear()
        {
            value = 0;
            tsec = 0;
        }
    };


    class SystemControlResources
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        //The reference to the manager is used primarily for
        //setup purposes.
        Rabit::RabitManager* _mgrPtr;

    public:

        DataRecorderStdHeader DataRecorderStdHdr;

        VidereSystemControlDataRecord SystemControlDataRecord;


        //Messages:
        //The Head Orientation relative to the Vehicle... used to control
        //the vehicle.
        std::shared_ptr<QuaternionMessage> HeadOrientationQuaternionMsg;

        std::shared_ptr<QuaternionMessage> VehicleOrientationQuaternionMsg;

        std::shared_ptr<SipnPuffMessage> SipnPuffMsg;

        std::shared_ptr<GPSFixMessage> GPSFixMsgPtr;

        std::shared_ptr<VehicleControlParametersMessage> VehicleControlParametersMsg;

        std::shared_ptr<SipnPuffControlMessage> SipnPuffCtrlMsg;

        std::shared_ptr<ResetOrientationStateMessage> ResetOrientationStateMsg;


        std::shared_ptr<VidereSystemControlMessage> VidereSystemCtrlMsg;
        std::shared_ptr<VidereSystemControlMessage> VidereSystemCtrlStatusMsg;

        //Steering Control Messages
        std::shared_ptr<SteeringTorqueCtrlMessage> SteeringTorqueCtrlFromVidereMonitorMsg;
        std::shared_ptr<SteeringTorqueCtrlMessage> SteeringTorqueCtrlMsg;
        std::shared_ptr<DceEPASteeringStatusMessage> SteeringStatusMsg;

        std::shared_ptr<LinearActuatorPositionCtrlMessage> BrakeControlFromVidereMonitorMsg;
        std::shared_ptr<LinearActuatorPositionCtrlMessage> BrakeControlMsg;
        std::shared_ptr<LinearActuatorPositionCtrlMessage> BrakeControlFeedbackMsg;

        std::shared_ptr<LinearActuatorPositionCtrlMessage> ThrottleControlFromVidereMonitorMsg;
        std::shared_ptr<LinearActuatorPositionCtrlMessage> ThrottlePositionControlMsg;
        std::shared_ptr<LinearActuatorPositionCtrlMessage> ThrottleControlFeedbackMsg;

        std::shared_ptr<VehicleSwitchInputMessage> VehicleSwitchInputMsg;

        std::shared_ptr<RemoteControlInputMessage> RemoteControlInputMsg;

        std::shared_ptr<ImageLoggingControlMessage> LoggingControlMsg;

        std::shared_ptr<StreamRecordImageControlMesssage> StreamRecordImageControlMsg;

        std::shared_ptr<ImageProcessControlMessage> ImageProcessControlMsg;

        std::shared_ptr<ImageCaptureControlMessage> ImageCaptureControlMsg;

        std::shared_ptr<HeadOrientationControlMessage> HeadOrientationControlMsg;


        //Remote

        ManualDriverControlState_e MDCState = ManualDriverControlState_e::MDCS_Reset;

        //Steering Torque Map... Set with config
        int EPASTorqueMapNo = 3;

        double ControlLoopRunRateHz = 100.0;
        int ControlLoopDelaymsec = 10;

        double CurrentTimeStamp = 0;
        double LastTimeStamp = 0;
        double DeltaTimeSec = 0;

        static const int SAHistoryLen = 3;

    private:


        ValueAtTime_t _epasSteeringAngleHistory[SAHistoryLen];
        ValueAtTime_t _steeringAngleErrorHistory[SAHistoryLen];

        double _steeringAngleFBIntegralVal = 0;

        void clearSteeringAngleHistory();

        void addNewEpasSteeringAngle(double angle, double tsec);
        void addNewSteeringAngleError(double angle, double tsec);


    public:
        SystemControlResources(Rabit::RabitManager* mgrPtr,
                               std::shared_ptr<ConfigData> config);

        ~SystemControlResources();

        void Initialize();

        std::shared_ptr<ConfigData> GetConfig()
        {
            return _config_sptr;
        }

        Rabit::RabitManager* GetMgrPtr()
        {
            return _mgrPtr;
        }

        double SetCurrentTimestamp();

        void ResetSteeringAngleFeedbackLoop();

        double SteeringAngleFeedbackLoop(double headAngleDegrees, double dt);


    };

}
#endif //VIDERE_DEV_SYSTEMCONTROLRESOURCES_H
