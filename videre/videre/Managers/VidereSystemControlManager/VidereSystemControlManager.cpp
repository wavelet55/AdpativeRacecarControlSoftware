/* ****************************************************************
 * Athr(s): Randy Direen, PhD
 * Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Aug, 2018
 *
 * Videre System Control Manager
 * This is the top-level system control of the racecar
 *******************************************************************/


#include "VidereSystemControlManager.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

namespace videre
{

    VidereSystemControlManager::VidereSystemControlManager(std::string name,
                                     std::shared_ptr<ConfigData> config)
              : VidereSystemControlManagerWSRMgr(name),
                SCR(this, config),
                _dataRecorder(),
              _hovcProcess(this, config, &SCR)
    {
        SetWakeupTimeDelayMSec(10);
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        //Messages

        _dataRecorder.setDirectory(DataLogDirectory);
        string fn = config->GetConfigStringValue("VidereSystemControl.DataLogBaseFilename", "VidereSystemControlDataLog");
        _dataRecorder.setBaseFilename(fn);
        _dataRecorder.setHeaderRecord(&SCR.DataRecorderStdHdr);

        EnableVidereSystemControlfLogging = config->GetConfigBoolValue("VidereSystemControl.EnableLogging", true);

        //This flag when true by-passes the Driver Safety Switch...
        //it is only meant to be used for test when there is no extern
        //switch connected to the system.
        TestIgnoreDriverSw = false;
        TestIgnoreDriverSw = config->GetConfigBoolValue("VidereSystemControl.TestIgnoreDriverSw", false);

        _enableLoggingDuringHeadControl = config->GetConfigBoolValue("VidereSystemControl.AutoEnableLoggingInHeadControl", false);

        _startupDefaultSystemState = VidereSystemStates_e::VSS_HeadOrientationControl;
        string startupStateStr = config->GetConfigStringValue("VidereSystemControl.StartupState", "HC");
        if(startupStateStr.c_str()[0] == 'R' || startupStateStr.c_str()[0] == 'r')
        {
            _startupDefaultSystemState = VidereSystemStates_e::VSS_RemoteControl;
        }
    }

    void VidereSystemControlManager::Initialize()
    {
        LOGINFO("VidereSystemControlManager: Initialization Started");

        SCR.Initialize();

        double runRateHz = _config_sptr->GetConfigDoubleValue("HOVehicleControl.SystemControlMgrUpdateRateHz", 100.0);
        runRateHz = runRateHz < 10 ? 10 : runRateHz > 1000 ? 1000 : runRateHz;
        int delayms = (int)( (1000.0 / runRateHz) + 0.5 );

        SetWakeupTimeDelayMSec(delayms);
        try
        {
            ResetSubStates();
            //_stopWatch.reset();
        }
        catch(exception e)
        {
            LOGERROR("VidereSystemControlManager: Event open exception: " << e.what());
        }

    }

    void VidereSystemControlManager::Startup()
    {
        _hovcProcess.Initialize();
        _gpio2DriverEnalbeDebounceCntr = 0;
        _gpio4AuxEnalbeDebounceCntr = 0;

        try
        {
            string fn = _config_sptr->GetConfigStringValue("RudiTX2_GPIO.GPIO2_Input_Fn", "/sys/class/gpio/gpio233/value");
            _gpio2DriverEnalbeInput = fopen(fn.c_str(), "r");
            if(_gpio2DriverEnalbeInput == nullptr)
            {
                LOGERROR("VidereSystemControlManager: Error Setting up GPIO-2 Input.");
            }
            else
            {
                setvbuf( _gpio2DriverEnalbeInput, (char *)NULL, _IONBF, 0 );
            }

            //GPIO1 must have a 1 written to it... it is being used for a pull-up voltate for GPIO-2
            fn = _config_sptr->GetConfigStringValue("RudiTX2_GPIO.GPIO1_Output_Fn", "/sys/class/gpio/gpio232/value");
            _gpio1PullupOutput = fopen(fn.c_str(), "wb");
            if(_gpio1PullupOutput == nullptr)
            {
                LOGERROR("VidereSystemControlManager: Error Setting up GPIO-1 Output.");
            }
            else
            {
                char outval = 1;
                setvbuf( _gpio1PullupOutput, (char *)NULL, _IONBF, 0 );
                //Write a 1 to the output to pull it up to 3.3 volts
                fwrite(&outval, 1, 1, _gpio2DriverEnalbeInput);
            }


            fn = _config_sptr->GetConfigStringValue("RudiTX2_GPIO.GPIO4_Input_Fn", "/sys/class/gpio/gpio235/value");
            _gpio4AuxEnalbeInput = fopen(fn.c_str(), "r");
            if(_gpio4AuxEnalbeInput == nullptr)
            {
                LOGERROR("VidereSystemControlManager: Error Setting up GPIO-4 Input.");
            }
            else
            {
                setvbuf( _gpio4AuxEnalbeInput, (char *)NULL, _IONBF, 0 );
            }
        }
        catch(exception e)
        {
            LOGERROR("VidereSystemControlManager: Error Setting up GPIO Input: " << e.what());
        }

    }

    void VidereSystemControlManager::Shutdown()
    {
        _dataRecorder.closeLogFile();
        if(_gpio2DriverEnalbeInput != nullptr )
        {
            fclose(_gpio2DriverEnalbeInput);
        }
        if(_gpio1PullupOutput != nullptr )
        {
            fclose(_gpio1PullupOutput);
        }

        if(_gpio4AuxEnalbeInput != nullptr )
        {
            fclose(_gpio4AuxEnalbeInput);
        }
        LOGINFO("VidereSystemControlManager shutdown");
    }

    //At this point in time there is only 1 switch we care about... so just read the one
    //switch.
    void VidereSystemControlManager::ReadVehicleInputSwitches()
    {
        bool swState = false;
        char inpVal;
        try
        {
            if(_gpio2DriverEnalbeInput != nullptr)
            {
                fseek(_gpio2DriverEnalbeInput, 0L, SEEK_SET);
                fread(&inpVal, 1, 1, _gpio2DriverEnalbeInput);
                swState = inpVal == '1';
                if( SCR.VehicleSwitchInputMsg->DriverControlEnabled != swState)
                {
                    if( ++_gpio2DriverEnalbeDebounceCntr > 2)
                    {
                        SCR.VehicleSwitchInputMsg->DriverControlEnabled = swState;
                        SCR.VehicleSwitchInputMsg->PostMessage();
                        _gpio2DriverEnalbeDebounceCntr = 0;
                        if(swState)
                        {
                            LOGINFO("Driver Control Switch Enabled.");
                        }
                        else
                        {
                            LOGINFO("Driver Control Switch Disabled.");
                        }
                        if(_enableLoggingDuringHeadControl)
                        {
                            SCR.LoggingControlMsg->FetchMessage();
                            SCR.LoggingControlMsg->EnableLogging = swState;
                            SCR.LoggingControlMsg->PostMessage();

                            SCR.StreamRecordImageControlMsg->FetchMessage();
                            SCR.StreamRecordImageControlMsg->RecordImagesEnabled = swState;
                            SCR.StreamRecordImageControlMsg->PostMessage();
                        }
                    }
                }
                else
                {
                    _gpio2DriverEnalbeDebounceCntr = 0;
                }
            }

            SCR.SystemControlDataRecord.DriverEnableSW = (uint8_t)SCR.VehicleSwitchInputMsg->DriverControlEnabled == true ? 1 : 0;
        }
        catch(exception e)
        {
            LOGERROR("VidereSystemControlManager: Error Reading GPIO Input: " << e.what());
        }

    }

    //Returns true if the system needs to switch to safety driver control.
    bool VidereSystemControlManager::CheckForSwitchToSafetyDriverControl()
    {
        bool switchToSafetyDriverControl = false;
        SCR.SteeringStatusMsg->FetchMessage();
        if( (!SCR.VehicleSwitchInputMsg->DriverControlEnabled && !TestIgnoreDriverSw)
           || SCR.SteeringStatusMsg->DriverTorqueHit)
        {
            ResetVehicleActuators();
            ResetSubStates();
            _systemState = VidereSystemStates_e::VSS_ManualDriverControl;
            LOGINFO("Exiting Driver Head Orientation Control.");
            if(SCR.SteeringStatusMsg->DriverTorqueHit)
            {
                SCR.SystemControlDataRecord.DriverTorqueHit = 1;
                LOGWARN("Safety Driver Took control of Steering Wheel.");
            }
            switchToSafetyDriverControl = true;
        }
        return switchToSafetyDriverControl;
    }

    //Note: RemoteControl and  HeadOrientationControl divert to the ManualDriverControl
    //state when the VehicleInputSwitch Driver Enable is disabled/Off or if there
    //is excess torque on the steering wheel.  This give the safety driver the means to
    //take control of the car.
    void VidereSystemControlManager::ExecuteUnitOfWork()
    {
        SystemStateResponse_t response;
        bool updateStatusMsg = false;
        SetWakeupTimeDelayMSec(SCR.ControlLoopDelaymsec);

        //Each State/Process is responsible for filling in the appropiate Data Log Values
        SCR.SystemControlDataRecord.Clear();

        ReadVehicleInputSwitches();
        SCR.VidereSystemCtrlMsg->FetchMessage();

        SCR.SetCurrentTimestamp();
        SCR.SteeringStatusMsg->FetchMessage();
        SCR.SystemControlDataRecord.SteeringAngle = SCR.SteeringStatusMsg->SteeringAngleDeg;
        SCR.SystemControlDataRecord.ControlTypeState = (uint8_t)_systemState;

        switch(_systemState)
        {
            case VidereSystemStates_e::VSS_Init:
                response = InitState();
                if(response.Response == SystemStateResponse_e::SSR_ChangeStateRequest )
                {
                    ResetSubStates();
                    ResetVehicleActuators();
                    SCR.ResetOrientationStateMsg->ResetHeadOrientationState = true;
                    SCR.ResetOrientationStateMsg->ResetVehicleOrientationState = true;
                    SCR.ResetOrientationStateMsg->PostMessage();
                    _systemState = response.NextDesiredState;
                    updateStatusMsg = true;
                }
                break;

            case VidereSystemStates_e::VSS_RemoteControl:
                if(CheckForSwitchToSafetyDriverControl())
                {
                    _systemState = VidereSystemStates_e::VSS_ManualDriverControl;
                    SCR.ResetOrientationStateMsg->ResetHeadOrientationState = true;
                    SCR.ResetOrientationStateMsg->ResetVehicleOrientationState = true;
                    SCR.ResetOrientationStateMsg->PostMessage();
                    updateStatusMsg = true;
                }
                else
                {
                    response = RemoteControlState();
                    if(SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_ExternalMonitorControl
                       || SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_ManualDriverControl
                       || SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_HeadOrientationCal)
                    {
                        ResetVehicleActuators();
                        ResetSubStates();
                        _systemState = SCR.VidereSystemCtrlMsg->SystemState;
                        updateStatusMsg = true;
                    }
                    else if(SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_HeadOrientationControl)
                    {
                        ResetVehicleActuators();
                        ResetSubStates();
                        //Force going to the Safety Driver Control state first... that
                        //state will switch to Head Orientation or Remote control once the
                        //Driver enable switch is disabled then enabled.
                        _systemState = VidereSystemStates_e::VSS_ManualDriverControl;
                        updateStatusMsg = true;
                    }
                }
                break;

            case VidereSystemStates_e::VSS_ExternalMonitorControl:
                response = ExternalMonitorControlState();
                if( SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_ManualDriverControl
                   || SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_HeadOrientationCal )
                {
                    ResetVehicleActuators();
                    ResetSubStates();
                    _systemState = SCR.VidereSystemCtrlMsg->SystemState;
                    updateStatusMsg = true;
                }
                else if( SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_HeadOrientationControl
                         || SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_RemoteControl)
                {
                    ResetVehicleActuators();
                    ResetSubStates();
                    //Force going to the Safety Driver Control state first... that
                    //state will switch to Head Orientation or Remote control once the
                    //Driver enable switch is disabled then enabled.
                    _systemState = VidereSystemStates_e::VSS_ManualDriverControl;
                    updateStatusMsg = true;
                }

                break;

            case VidereSystemStates_e::VSS_ManualDriverControl:
                response = ManualDriverControlState();
                if( SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_ExternalMonitorControl
                   || SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_HeadOrientationCal )
                {
                    ResetVehicleActuators();
                    ResetSubStates();
                    _systemState = SCR.VidereSystemCtrlMsg->SystemState;
                    updateStatusMsg = true;
                }
                else if(response.Response == SystemStateResponse_e::SSR_ChangeStateRequest )
                {
                    ResetVehicleActuators();
                    ResetSubStates();
                    SCR.ResetOrientationStateMsg->ResetHeadOrientationState = true;
                    SCR.ResetOrientationStateMsg->ResetVehicleOrientationState = true;
                    SCR.ResetOrientationStateMsg->PostMessage();
                    _systemState = response.NextDesiredState;
                    _systemState = response.NextDesiredState;
                    updateStatusMsg = true;
                }

                break;

            case VidereSystemStates_e::VSS_HeadOrientationCal:

                if(SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_ExternalMonitorControl
                   || SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_ManualDriverControl )
                {
                    ResetVehicleActuators();
                    ResetSubStates();
                    _systemState = SCR.VidereSystemCtrlMsg->SystemState;
                    updateStatusMsg = true;
                }
                else if( SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_HeadOrientationControl
                         || SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_RemoteControl)
                {
                    ResetVehicleActuators();
                    ResetSubStates();
                    //Force going to the Safety Driver Control state first... that
                    //state will switch to Head Orientation or Remote control once the
                    //Driver enable switch is disabled then enabled.
                    _systemState = VidereSystemStates_e::VSS_ManualDriverControl;
                    updateStatusMsg = true;
                }
                break;

            case VidereSystemStates_e::VSS_HeadOrientationControl:
                if(CheckForSwitchToSafetyDriverControl())
                {
                    _systemState = VidereSystemStates_e::VSS_ManualDriverControl;
                    SCR.ResetOrientationStateMsg->ResetHeadOrientationState = true;
                    SCR.ResetOrientationStateMsg->ResetVehicleOrientationState = true;
                    SCR.ResetOrientationStateMsg->PostMessage();
                    updateStatusMsg = true;
                }
                else
                {
                    response = _hovcProcess.ExecuteUnitOfWork();
                    if(response.Response == SystemStateResponse_e::SSR_ChangeStateRequest)
                    {
                        ResetVehicleActuators();
                        ResetSubStates();
                        SCR.ResetOrientationStateMsg->ResetHeadOrientationState = true;
                        SCR.ResetOrientationStateMsg->ResetVehicleOrientationState = true;
                        SCR.ResetOrientationStateMsg->PostMessage();
                        _systemState = response.NextDesiredState;
                        updateStatusMsg = true;
                    }
                    else if( SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_ExternalMonitorControl
                       || SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_HeadOrientationCal
                       || SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_ManualDriverControl)
                    {

                        ResetVehicleActuators();
                        ResetSubStates();
                        _systemState = SCR.VidereSystemCtrlMsg->SystemState;
                        updateStatusMsg = true;
                    }
                }
                break;

        }

        SCR.VidereSystemCtrlStatusMsg->SystemState = _systemState;
        SCR.VidereSystemCtrlStatusMsg->HeadControlEnable = SCR.VidereSystemCtrlMsg->HeadControlEnable;
        SCR.VidereSystemCtrlStatusMsg->ThrottleControlEnable = SCR.VidereSystemCtrlMsg->ThrottleControlEnable;
        SCR.VidereSystemCtrlStatusMsg->BrakeControlEnable = SCR.VidereSystemCtrlMsg->BrakeControlEnable;
        SCR.VidereSystemCtrlStatusMsg->BCIControlEnable = SCR.VidereSystemCtrlMsg->BCIControlEnable;

        SCR.VidereSystemCtrlStatusMsg->StatusCounter = ++_ctrlLoopCount;
        if( updateStatusMsg || _ctrlLoopCount % 100 == 0)
        {
            SCR.VidereSystemCtrlStatusMsg->PostMessage();
        }


        bool logMsgChanged = SCR.LoggingControlMsg->FetchMessage();
        if( EnableVidereSystemControlfLogging && SCR.LoggingControlMsg->EnableLogging)
        {
            _dataRecorder.writeDataRecord(SCR.SystemControlDataRecord);
        }
        else if(logMsgChanged && !SCR.LoggingControlMsg->EnableLogging)
        {
            _dataRecorder.closeLogFile();
        }


    }

    void VidereSystemControlManager::ResetSubStates()
    {
        _hovcProcess.Reset();

        //When ever we enter the ManualDriverControlState, it must start in
        //its own reset state.
        SCR.MDCState = ManualDriverControlState_e::MDCS_Reset;
    }

    //This routine should be called when switching states to ensure the
    //Brake, Throttle, and Steering actuators are disable and not left in
    //some running-actuated state.
    void VidereSystemControlManager::ResetVehicleActuators()
    {
        SCR.BrakeControlMsg->Clear();
        SCR.BrakeControlMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
        SCR.BrakeControlMsg->PostMessage();

        //Return Throttle actuator to the default state.
        SCR.ThrottlePositionControlMsg->Clear();
        SCR.ThrottlePositionControlMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
        SCR.ThrottlePositionControlMsg->PostMessage();

        //Return Steering actuator to the default state.
        SCR.SteeringTorqueCtrlMsg->Clear();
        SCR.SteeringTorqueCtrlMsg->PostMessage();

        SCR.ResetSteeringAngleFeedbackLoop();
    }

    SystemStateResponse_t VidereSystemControlManager::InitState()
    {
        SystemStateResponse_t response;
        _ctrlLoopCount = 0;
        ResetVehicleActuators();

        //Setup and enable Image processing
        SCR.ImageProcessControlMsg->GPUProcessingEnabled = false;
        SCR.ImageProcessControlMsg->VisionProcessingMode = VisionProcessingMode_e::VisionProcMode_HeadTrackingProc;
        SCR.ImageProcessControlMsg->TargetProcessingMode = TargetProcessingMode_e::TgtProcMode_Std;
        SCR.ImageProcessControlMsg->TargetImageProcessingEnabled = true;
        SCR.ImageProcessControlMsg->PostMessage();

        SCR.ImageCaptureControlMsg->Clear();
        SCR.ImageCaptureControlMsg->ImageCaptureEnabled = true;
        SCR.ImageCaptureControlMsg->PostMessage();

        SCR.VidereSystemCtrlMsg->Clear();
        SCR.VidereSystemCtrlMsg->SystemState = _startupDefaultSystemState;
        SCR.VidereSystemCtrlMsg->ThrottleControlEnable = _config_sptr->GetConfigBoolValue("VidereSystemControl.ThrottleControlEnable", true);
        SCR.VidereSystemCtrlMsg->BrakeControlEnable = _config_sptr->GetConfigBoolValue("VidereSystemControl.BrakeControlEnable", true);
        SCR.VidereSystemCtrlMsg->HeadControlEnable = _config_sptr->GetConfigBoolValue("VidereSystemControl.HeadControlEnable", true);
        SCR.VidereSystemCtrlMsg->BCIControlEnable = _config_sptr->GetConfigBoolValue("BCI_Throttle_Control.EnableNexusBCIThrottleControl", false);
        SCR.VidereSystemCtrlMsg->PostMessage();

        WorkSpace()->BCIControlConfigMsg->EnableNexusBCIThrottleControl = _config_sptr->GetConfigBoolValue("BCI_Throttle_Control.EnableNexusBCIThrottleControl", false);
        WorkSpace()->BCIControlConfigMsg->SipnPuffBrakeOnlyControl = _config_sptr->GetConfigBoolValue("BCI_Throttle_Control.SipnPuffBrakeOnlyControl", true);
        WorkSpace()->BCIControlConfigMsg->BCIThrottleIntegrationGain = _config_sptr->GetConfigDoubleValue("BCI_Throttle_Control.BCIThrottleIntegrationGain", 1.0);
        WorkSpace()->BCIControlConfigMsg->BCIThrottleRampDownIntegrationGain = _config_sptr->GetConfigDoubleValue("BCI_Throttle_Control.BCIThrottleRampDownIntegrationGain", 0.25);
        WorkSpace()->BCIControlConfigMsg->BCIThrottleRampDownDelaySeconds = _config_sptr->GetConfigDoubleValue("BCI_Throttle_Control.BCIThrottleRampDownDelaySeconds", 2.5);
        WorkSpace()->BCIControlConfigMsg->PostMessage();

        WorkSpace()->BCIControlConfigFeedbackMsg->CopyMessage(WorkSpace()->BCIControlConfigMsg.get());
        WorkSpace()->BCIControlConfigFeedbackMsg->PostMessage();


        if(SCR.VidereSystemCtrlMsg->BCIControlEnable)
        {
            LOGINFO("Nexus BCI Throttle Control is Enabled!");
            cout << "Nexus BCI Throttle Control is Enabled!" << endl;
        }

        SCR.LoggingControlMsg->FetchMessage();

        response.Response = SystemStateResponse_e::SSR_ChangeStateRequest;
        response.NextDesiredState = VidereSystemStates_e::VSS_HeadOrientationControl;
        SCR.MDCState = ManualDriverControlState_e::MDCS_Reset;
        return response;
    }

    //In this state the game-remote control input is piped to
    //controlling the car actuators.
    SystemStateResponse_t VidereSystemControlManager::RemoteControlState()
    {
        SystemStateResponse_t response;
        SCR.VehicleControlParametersMsg->FetchMessage();
        if( SCR.RemoteControlInputMsg->FetchMessage() )
        {
            double throttleCtrl = 0;
            double brakeCtrl = 0;
            double tbVal = SCR.RemoteControlInputMsg->ThrottleBrakePercent;
            if( tbVal >= 0 )
            {
                throttleCtrl = tbVal;
                SCR.ThrottlePositionControlMsg->ClutchEnable = true;
                SCR.ThrottlePositionControlMsg->MotorEnable = true;
                SCR.ThrottlePositionControlMsg->ManualExtControl = true;
                SCR.ThrottlePositionControlMsg->setPositionPercent(throttleCtrl);
                SCR.ThrottlePositionControlMsg->PostMessage();

                SCR.BrakeControlMsg->Clear();
                SCR.BrakeControlMsg->PostMessage();
            }
            else
            {
                brakeCtrl = -tbVal;
                throttleCtrl = tbVal;
                SCR.BrakeControlMsg->ClutchEnable = true;
                SCR.BrakeControlMsg->MotorEnable = true;
                SCR.BrakeControlMsg->ManualExtControl = true;
                SCR.BrakeControlMsg->setPositionPercent(brakeCtrl);
                SCR.BrakeControlMsg->PostMessage();

                SCR.ThrottlePositionControlMsg->Clear();
                SCR.ThrottlePositionControlMsg->PostMessage();
            }

            //Control the Steering
            SCR.SteeringTorqueCtrlMsg->ManualExtControl = false;
            SCR.SteeringTorqueCtrlMsg->SteeringControlEnabled = true;
            SCR.SteeringTorqueCtrlMsg->setSteeringTorqueMap(SCR.EPASTorqueMapNo);

            double steeringTorqueControlPercent = 0;
            if(SCR.VehicleControlParametersMsg->UseSteeringAngleControl)
            {
                double deltaTimeSec = SCR.DeltaTimeSec;
                //ensure a reasonable time... there could have been a brake in action
                //or a new start.
                deltaTimeSec = deltaTimeSec < 0.001 ? 0.001 : deltaTimeSec > 0.1 ? 0.1 : deltaTimeSec;
                double rcCtrlAngle = 0.25 * SCR.RemoteControlInputMsg->SteeringControlPercent;
                rcCtrlAngle = SCR.VehicleControlParametersMsg->RCSteeringGain * rcCtrlAngle;
                steeringTorqueControlPercent = SCR.SteeringAngleFeedbackLoop(rcCtrlAngle, deltaTimeSec);
            }
            else
            {
                steeringTorqueControlPercent = SCR.RemoteControlInputMsg->SteeringControlPercent;
                steeringTorqueControlPercent *= SCR.VehicleControlParametersMsg->SteeringControlGain;
            }

            //The RemoteControlInputMsg->SteeringControlPercent is in the range: [-100, 100]
            SCR.SteeringTorqueCtrlMsg->setSteeringTorquePercent(steeringTorqueControlPercent);
            SCR.SteeringTorqueCtrlMsg->PostMessage();

        }
        return response;
    }


    //This state is used for control coming from the Videre Monitor program.
    //It is basically a manual control state.
    //This is the default state of the system until instructed to go to another
    //state.
    SystemStateResponse_t VidereSystemControlManager::ExternalMonitorControlState()
    {
        SystemStateResponse_t response;
        response.Clear();

        SCR.VehicleControlParametersMsg->FetchMessage();

        //The Monitor program can send Brake, Throttle and Steering control
        //to the actuators.  This is primarily for test and should only be done
        //in well controlled situation.
        if( SCR.BrakeControlFromVidereMonitorMsg->FetchMessage() )
        {
            if( SCR.BrakeControlFromVidereMonitorMsg->ManualExtControl)
            {
                SCR.BrakeControlMsg->CopyMessage(SCR.BrakeControlFromVidereMonitorMsg.get());
                SCR.BrakeControlMsg->PostMessage();
            }
            else if(SCR.BrakeControlFromVidereMonitorMsg->ActuatorSetupMode)
            {
                SCR.BrakeControlMsg->Clear();
                SCR.BrakeControlMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
                SCR.BrakeControlMsg->ActuatorSetupMode = true;
                SCR.BrakeControlMsg->PostMessage();
            }
            else
            {
                //Return Brake actuator to the default state.
                SCR.BrakeControlMsg->Clear();
                SCR.BrakeControlMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
                SCR.BrakeControlMsg->PostMessage();
            }
        }

        if( SCR.ThrottleControlFromVidereMonitorMsg->FetchMessage() )
        {
            if( SCR.ThrottleControlFromVidereMonitorMsg->ManualExtControl)
            {
                SCR.ThrottlePositionControlMsg->CopyMessage(SCR.ThrottleControlFromVidereMonitorMsg.get());
                SCR.ThrottlePositionControlMsg->PostMessage();
            }
            else if(SCR.ThrottleControlFromVidereMonitorMsg->ActuatorSetupMode)
            {
                SCR.ThrottlePositionControlMsg->Clear();
                SCR.ThrottlePositionControlMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
                SCR.ThrottlePositionControlMsg->ActuatorSetupMode = true;
                SCR.ThrottlePositionControlMsg->PostMessage();
            }
            else
            {
                //Return Throttle actuator to the default state.
                SCR.ThrottlePositionControlMsg->Clear();
                SCR.ThrottlePositionControlMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
                SCR.ThrottlePositionControlMsg->PostMessage();
            }
        }

        if( SCR.SteeringTorqueCtrlFromVidereMonitorMsg->FetchMessage() )
        {
            if( SCR.SteeringTorqueCtrlFromVidereMonitorMsg->ManualExtControl)
            {
                SCR.SteeringTorqueCtrlMsg->CopyMessage(SCR.SteeringTorqueCtrlFromVidereMonitorMsg.get());
                SCR.SteeringTorqueCtrlMsg->PostMessage();
            }
            else
            {
                //Return Steering actuator to the default state.
                SCR.SteeringTorqueCtrlMsg->Clear();
                SCR.SteeringTorqueCtrlMsg->PostMessage();
            }
        }

        return response;
    }


    //This is the state when in the car and getting ready for Head Orientation
    //control or when the safety driver takes back over the car.
    SystemStateResponse_t VidereSystemControlManager::ManualDriverControlState()
    {
        SystemStateResponse_t response;
        if(TestIgnoreDriverSw)
        {
            if( SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_HeadOrientationControl
                || SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_RemoteControl)
            {
                ResetVehicleActuators();
                LOGINFO("Driver Control Switch Enabled... Going to Driver or Remote Control.")
                response.NextDesiredState = SCR.VidereSystemCtrlMsg->SystemState;
                response.Response = SystemStateResponse_e::SSR_ChangeStateRequest;
            }
        }
        else
        {
            switch(SCR.MDCState)
            {
                case ManualDriverControlState_e::MDCS_Reset:
                    //Before we can enable the Driver controll... the
                    //driver control switch must start in the off/disabled stated.
                    //This keeps the system from jumping back into the driver control state
                    //if an error occurs and then clears... such as stearing over-torque.
                    if(SCR.VehicleSwitchInputMsg->DriverControlEnabled)
                    {
                        LOGWARN("Entered Manual Driver Control State with the Driver Control SW Enabled.")
                        SCR.MDCState = ManualDriverControlState_e::MDCS_DriverControlSWEnabled;
                    } else
                    {
                        SCR.MDCState = ManualDriverControlState_e::MDCS_DriverControlSWDisabled;
                    }
                    break;

                case ManualDriverControlState_e::MDCS_DriverControlSWDisabled:
                    if( SCR.VehicleSwitchInputMsg->DriverControlEnabled
                       && (SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_HeadOrientationControl
                           || SCR.VidereSystemCtrlMsg->SystemState == VidereSystemStates_e::VSS_RemoteControl))
                    {
                        ResetVehicleActuators();
                        LOGINFO("Driver Control Switch Enabled... Going to Driver or Remote Control.")
                        response.NextDesiredState = SCR.VidereSystemCtrlMsg->SystemState;
                        response.Response = SystemStateResponse_e::SSR_ChangeStateRequest;
                    }
                    break;

                case ManualDriverControlState_e::MDCS_DriverControlSWEnabled:
                    //Wait in this state until the driver control switch is disabled.
                    if(!SCR.VehicleSwitchInputMsg->DriverControlEnabled)
                    {
                        SCR.MDCState = ManualDriverControlState_e::MDCS_DriverControlSWDisabled;
                    }
                    break;
            }
        }
        return response;
    }



}