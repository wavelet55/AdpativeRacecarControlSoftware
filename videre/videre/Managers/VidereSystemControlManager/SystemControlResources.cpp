/* ****************************************************************
 * System Control Resources
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug. 2018
 *
  *******************************************************************/

#include "SystemControlResources.h"

using namespace std;

namespace videre
{

    SystemControlResources::SystemControlResources(Rabit::RabitManager* mgrPtr,
                                                   std::shared_ptr<ConfigData> config)
        :   SystemControlDataRecord(), DataRecorderStdHdr("Videre System Control Data Log", 0)

    {
        _mgrPtr = mgrPtr;
        _config_sptr = config;
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        double runRateHz = _config_sptr->GetConfigDoubleValue("HOVehicleControl.SystemControlMgrUpdateRateHz", 100.0);
        ControlLoopRunRateHz = runRateHz < 10 ? 10 : runRateHz > 1000 ? 1000 : runRateHz;
        ControlLoopDelaymsec = (int)( (1000.0 / ControlLoopRunRateHz) + 0.5 );

        //Messages
        VidereSystemCtrlMsg = std::make_shared<VidereSystemControlMessage>();
        VidereSystemCtrlStatusMsg = std::make_shared<VidereSystemControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("VidereSystemCtrlMsg", VidereSystemCtrlMsg);
        _mgrPtr->AddPublishSubscribeMessage("VidereSystemCtrlStatusMsg", VidereSystemCtrlStatusMsg);

        HeadOrientationQuaternionMsg = std::make_shared<QuaternionMessage>();
        _mgrPtr->AddPublishSubscribeMessage("HeadOrientationQuaternionMsg", HeadOrientationQuaternionMsg);
        //_mgrPtr->WakeUpManagerOnMessagePost(HeadOrientationQuaternionMsg);

        VehicleOrientationQuaternionMsg = make_shared<QuaternionMessage>();
        _mgrPtr->AddPublishSubscribeMessage("VehicleOrientationQuaternionMsg", VehicleOrientationQuaternionMsg);

        VehicleControlParametersMsg = make_shared<VehicleControlParametersMessage>();
        _mgrPtr->AddPublishSubscribeMessage("VehicleControlParametersMsg", VehicleControlParametersMsg);

        SipnPuffMsg = std::make_shared<SipnPuffMessage>();
        _mgrPtr->AddPublishSubscribeMessage("SipnPuffMessage", SipnPuffMsg);

        ResetOrientationStateMsg = std::make_shared<ResetOrientationStateMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ResetOrientationStateMessage", ResetOrientationStateMsg);

        SipnPuffCtrlMsg = std::make_shared<SipnPuffControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("SipnPuffControlMessage", SipnPuffCtrlMsg);

        HeadOrientationControlMsg = make_shared<HeadOrientationControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("HeadOrientationControlMessage", HeadOrientationControlMsg);

        GPSFixMsgPtr = std::make_shared<GPSFixMessage>();
        mgrPtr->AddPublishSubscribeMessage("GPSFixMessage", GPSFixMsgPtr);

        SteeringTorqueCtrlFromVidereMonitorMsg = std::make_shared<SteeringTorqueCtrlMessage>();
        SteeringTorqueCtrlMsg = std::make_shared<SteeringTorqueCtrlMessage>();
        SteeringStatusMsg = std::make_shared<DceEPASteeringStatusMessage>();
        _mgrPtr->AddPublishSubscribeMessage("SteeringTorqueCtrlFromVidereMonitorMsg", SteeringTorqueCtrlFromVidereMonitorMsg);
        _mgrPtr->AddPublishSubscribeMessage("SteeringTorqueCtrlMsg", SteeringTorqueCtrlMsg);
        _mgrPtr->AddPublishSubscribeMessage("SteeringStatusMsg", SteeringStatusMsg);


        BrakeControlFromVidereMonitorMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        BrakeControlMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        BrakeControlFeedbackMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("BrakeControlFromVidereMonitorMsg", BrakeControlFromVidereMonitorMsg);
        _mgrPtr->AddPublishSubscribeMessage("BrakeLAPositionControlMsg", BrakeControlMsg);
        _mgrPtr->AddPublishSubscribeMessage("BrakeLAPositionFeedbackMsg", BrakeControlFeedbackMsg);

        BrakeControlFromVidereMonitorMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
        BrakeControlMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
        BrakeControlFeedbackMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;

        ThrottleControlFromVidereMonitorMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        ThrottlePositionControlMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        ThrottleControlFeedbackMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ThrottleControlFromVidereMonitorMsg", ThrottleControlFromVidereMonitorMsg);
        _mgrPtr->AddPublishSubscribeMessage("ThrottleLAPositionControlMsg", ThrottlePositionControlMsg);
        _mgrPtr->AddPublishSubscribeMessage("ThrottleLAPositionFeedbackMsg", ThrottleControlFeedbackMsg);

        ThrottleControlFromVidereMonitorMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
        ThrottlePositionControlMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
        ThrottleControlFeedbackMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;

        RemoteControlInputMsg = std::make_shared<RemoteControlInputMessage>();
        _mgrPtr->AddPublishSubscribeMessage("RemoteControlInputMsg", RemoteControlInputMsg);

        VehicleSwitchInputMsg = std::make_shared<VehicleSwitchInputMessage>();
        _mgrPtr->AddPublishSubscribeMessage("VehicleSwitchInputMsg", VehicleSwitchInputMsg);


        LoggingControlMsg = std::make_shared<ImageLoggingControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageLoggingControlMessage", LoggingControlMsg);

        StreamRecordImageControlMsg = std::make_shared<StreamRecordImageControlMesssage>();
        _mgrPtr->AddPublishSubscribeMessage("StreamRecordImageControlMesssage", StreamRecordImageControlMsg);


        ImageProcessControlMsg = std::make_shared<ImageProcessControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageProcessControlMessage", ImageProcessControlMsg);

        ImageCaptureControlMsg = std::make_shared<ImageCaptureControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageCaptureControlMessage", ImageCaptureControlMsg);


        EPASTorqueMapNo = _config_sptr->GetConfigIntValue("VehicleActuatorInterface.EPASTorqueMapNo", 3);
        EPASTorqueMapNo = EPASTorqueMapNo < 1 ? 1 : EPASTorqueMapNo > 5 ? 5 : EPASTorqueMapNo;


    }

    SystemControlResources::~SystemControlResources()
    {

    }

    void SystemControlResources::Initialize()
    {
        MDCState = ManualDriverControlState_e::MDCS_Reset;

        VehicleControlParametersMsg->SipnPuffBlowGain = _config_sptr->GetConfigDoubleValue("HOVehicleControl.SipnPuffBlowGain", 1.0);
        VehicleControlParametersMsg->SipnPuffSuckGain = _config_sptr->GetConfigDoubleValue("HOVehicleControl.SipnPuffSuckGain", 1.0);
        VehicleControlParametersMsg->SipnPuffDeadBandPercent = _config_sptr->GetConfigDoubleValue("HOVehicleControl.SipnPuffDeadBandPercent", 1.0);
        VehicleControlParametersMsg->ThrottleSipnPuffGain = _config_sptr->GetConfigBoolValue("HOVehicleControl.ThrottleSipnPuffGain", false);
        VehicleControlParametersMsg->ThrottleSipnPuffGain = _config_sptr->GetConfigDoubleValue("HOVehicleControl.ThrottleSipnPuffGain", 1.0);
        VehicleControlParametersMsg->BrakeSipnPuffGain = _config_sptr->GetConfigDoubleValue("HOVehicleControl.BrakeSipnPuffGain", 1.0);

        VehicleControlParametersMsg->ThrottleBrakeHeadTiltEnable = _config_sptr->GetConfigBoolValue("HOVehicleControl.ThrottleBrakeHeadTiltEnable", false);
        VehicleControlParametersMsg->ThrottleBrakeHeadTiltForwardDeadbandDegrees = _config_sptr->GetConfigDoubleValue("HOVehicleControl.ThrottleBrakeHeadTiltForwardDeadbandDegrees", 15.0);
        VehicleControlParametersMsg->ThrottleBrakeHeadTiltBackDeadbandDegrees = _config_sptr->GetConfigDoubleValue("HOVehicleControl.ThrottleBrakeHeadTiltBackDeadbandDegrees", 10.0);
        VehicleControlParametersMsg->ThrottleHeadTiltGain = _config_sptr->GetConfigDoubleValue("HOVehicleControl.ThrottleHeadTiltGain", 1.0);
        VehicleControlParametersMsg->BrakeHeadTiltGain = _config_sptr->GetConfigDoubleValue("HOVehicleControl.BrakeHeadTiltGain", 1.0);

        VehicleControlParametersMsg->UseSteeringAngleControl = _config_sptr->GetConfigBoolValue("HOVehicleControl.UseSteeringAngleControl", false);
        VehicleControlParametersMsg->SteeringDeadband = _config_sptr->GetConfigDoubleValue("HOVehicleControl.SteeringDeadband", 2.5);
        VehicleControlParametersMsg->SteeringControlGain = _config_sptr->GetConfigDoubleValue("HOVehicleControl.SteeringControlGain", 1.0);
        VehicleControlParametersMsg->SteeringBiasAngleDegrees = _config_sptr->GetConfigDoubleValue("HOVehicleControl.SteeringBiasAngleDegrees", 0.0);
        VehicleControlParametersMsg->RCSteeringGain = _config_sptr->GetConfigDoubleValue("HOVehicleControl.RCSteeringGain", 0.25);

        VehicleControlParametersMsg->MaxLRHeadRotationDegrees = _config_sptr->GetConfigDoubleValue("HOVehicleControl.MaxLRHeadRotationDegrees", 60.0);

        VehicleControlParametersMsg->HeadLeftRighLPFOrder = _config_sptr->GetConfigIntValue("HOVehicleControl.HeadLeftRighLPFOrder", 2);
        VehicleControlParametersMsg->HeadLeftRighLPFCutoffFreqHz = _config_sptr->GetConfigDoubleValue("HOVehicleControl.HeadLeftRighLPFCutoffFreqHz", 5.0);

        VehicleControlParametersMsg->SteeringAngleFeedback_Kp = _config_sptr->GetConfigDoubleValue("HOVehicleControl.SteeringAngleFeedback_Kp", 1.0);
        VehicleControlParametersMsg->SteeringAngleFeedback_Kd = _config_sptr->GetConfigDoubleValue("HOVehicleControl.SteeringAngleFeedback_Kd", 0.0);
        VehicleControlParametersMsg->SteeringAngleFeedback_Ki = _config_sptr->GetConfigDoubleValue("HOVehicleControl.SteeringAngleFeedback_Ki", 0.0);

        VehicleControlParametersMsg->PostMessage();

        //Push the config value over to the Sip and Puff manager.
        SipnPuffCtrlMsg->EnableSipnPuffIntegration = false;
        SipnPuffCtrlMsg->SipnPuffDeadBandPercent = VehicleControlParametersMsg->SipnPuffDeadBandPercent;
        SipnPuffCtrlMsg->SipnPuffBlowGain = VehicleControlParametersMsg->SipnPuffBlowGain;
        SipnPuffCtrlMsg->SipnPuffSuckGain = VehicleControlParametersMsg->SipnPuffSuckGain;
        SipnPuffCtrlMsg->PostMessage();

        HeadOrientationControlMsg->Clear();
        HeadOrientationControlMsg->SetHeadOrientation_QVar(_config_sptr->GetConfigDoubleValue("HOVehicleControl.HeadOrientation_QVar", 0.001));
        HeadOrientationControlMsg->SetHeadOrientation_RVar(_config_sptr->GetConfigDoubleValue("HOVehicleControl.HeadOrientation_RVar", 0.0003));
        HeadOrientationControlMsg->SetVehicleGravityFeedbackGain(_config_sptr->GetConfigDoubleValue("HOVehicleControl.SetVehicleGravityFeedbackGain", 0.99));
        HeadOrientationControlMsg->DisableVehicleInputToHeadOrientation = _config_sptr->GetConfigBoolValue("HOVehicleControl.DisableVehicleInputToHeadOrientation", false);
        HeadOrientationControlMsg->DisableVehicleGravityFeedback = _config_sptr->GetConfigBoolValue("HOVehicleControl.DisableVehicleGravityFeedback", false);
        HeadOrientationControlMsg->DisableHeadOrientationKalmanFilter = false;
        HeadOrientationControlMsg->HeadOrientationOutputSelect = HeadOrientationOutputSelect_e::HeadOrientation;
        HeadOrientationControlMsg->PostMessage();

    }

    double SystemControlResources::SetCurrentTimestamp()
    {
        LastTimeStamp = CurrentTimeStamp;
        CurrentTimeStamp = Rabit::SystemTimeClock::GetSystemTimeClock()->GetCurrentGpsTimeInSeconds();
        DeltaTimeSec = CurrentTimeStamp - LastTimeStamp;
        SystemControlDataRecord.VidereTimestamp = CurrentTimeStamp;
        SystemControlDataRecord.deltaTime = DeltaTimeSec;
    }

    void SystemControlResources::ResetSteeringAngleFeedbackLoop()
    {
        clearSteeringAngleHistory();
        _steeringAngleFBIntegralVal = 0;
        CurrentTimeStamp = 0;
        LastTimeStamp = 0;
        DeltaTimeSec = 0;
    }

    void SystemControlResources::clearSteeringAngleHistory()
    {
        for(int i = 0; i < SAHistoryLen; i++)
        {
            _epasSteeringAngleHistory[i].clear();
            _steeringAngleErrorHistory[i].clear();
        }
    }

    void SystemControlResources::addNewEpasSteeringAngle(double angle, double tsec)
    {
        for(int i = 1; i < SAHistoryLen; i++)
            _epasSteeringAngleHistory[i] = _epasSteeringAngleHistory[i - 1];

        _epasSteeringAngleHistory[0].value = angle;
        _epasSteeringAngleHistory[0].tsec = tsec;
    }

    void SystemControlResources::addNewSteeringAngleError(double angle, double tsec)
    {
        for(int i = 1; i < SAHistoryLen; i++)
            _steeringAngleErrorHistory[i] = _steeringAngleErrorHistory[i - 1];

        _steeringAngleErrorHistory[0].value = angle;
        _steeringAngleErrorHistory[0].tsec = tsec;
    }


    double SystemControlResources::SteeringAngleFeedbackLoop(double headAngleDegrees, double dt)
    {
        double epasSteeringAngle;
        double epasSATsec;
        double steeringTorquePercent = 0;


        //SteeringStatusMsg->FetchMessage();
        epasSteeringAngle = SteeringStatusMsg->SteeringAngleDeg;
        epasSATsec = SteeringStatusMsg->GetTimeStamp();
        addNewEpasSteeringAngle(epasSteeringAngle, epasSATsec);

        double saError = headAngleDegrees - epasSteeringAngle;
        //The torque control will be a function of the error.

        double fbVal = saError  * VehicleControlParametersMsg->SteeringAngleFeedback_Kp;

        //compute the approximate dirivative of the sa error... use a
        //weighted average over a few points in time.
        double derr = 0;
        double dte = CurrentTimeStamp - _steeringAngleErrorHistory[0].tsec;
        double loopdt = 0.001 * ControlLoopDelaymsec;
        if( _steeringAngleErrorHistory[0].tsec > 0 && dte >= loopdt && dte < 4 * loopdt )
        {
            derr = saError - _steeringAngleErrorHistory[0].value;
            derr = derr < -45.0 ? -45.0 : derr > 45.0 ? 45.0 : derr;
            derr = derr / dte;
        }
        addNewSteeringAngleError(saError, CurrentTimeStamp);
        fbVal += derr * VehicleControlParametersMsg->SteeringAngleFeedback_Kd;

        //Integrate the error... Limit the integrator to work on smaller angles so
        //that it does not have a big input on larger errors.
        double ierr = saError < -30 ? -30 : saError > 30 ? 30 : saError;
        ierr = fabs(ierr) < 2.0 ? 0 : ierr;
        _steeringAngleFBIntegralVal += dt * ierr;

        fbVal += ierr * VehicleControlParametersMsg->SteeringAngleFeedback_Ki;

        steeringTorquePercent = (100.0 / 60.0) * fbVal;

        SystemControlDataRecord.SAError = saError;
        SystemControlDataRecord.DtSAError = derr;
        SystemControlDataRecord.IntgSAError = ierr;

        return steeringTorquePercent;
    }





}
