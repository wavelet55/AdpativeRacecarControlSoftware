/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: May 8, 2018
 *
 * Sip-n-Puff Manager
 * Origin Instruments Breeze Sip/Puff
 * Ties into the Linux input system: /dev/input/eventx
 *******************************************************************/


#include "VehicleActuatorInterfaceManager.h"
#include <linux/input.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

#include <linux/can.h>
#include <linux/can/raw.h>
#include <boost/math/constants/constants.hpp>
#include <cmath>

using namespace std;

namespace videre
{

    VehicleActuatorInterfaceManager::VehicleActuatorInterfaceManager(std::string name,
                                     std::shared_ptr<ConfigData> config)
            : VehicleActuatorInterfaceManagerWSRMgr(name),
              _brakeActuator(this, LinearActuatorFunction_e::LA_Brake, config),
              _throttleActuator(this, LinearActuatorFunction_e::LA_Accelerator, config),
              _steeringControl(this, config), _canRxDataRecorder(), _canTxDataRecorder(),
              _dataRecorderCanRxHeader("Actuator CAN Bus Recieve Data Log", 0),
              _dataRecorderCanTxHeader("Actuator CAN Bus Transmit Data Log", 0),
              _brakePosCtrlRecord(), _throttlePosCtrlRecord(),
              _steeringCommandDataRecord(),
              _canRxDataLogReader()
    {
        this->SetWakeupTimeDelayMSec(1000);
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        _sipnPuffMsg = std::make_shared<SipnPuffMessage>();
        this->AddPublishSubscribeMessage("SipnPuffMessage", _sipnPuffMsg);
        //_sipnPuffMsg->Register_SomethingPublished(boost::bind(&VehicleActuatorInterfaceManager::WakeUpManagerEH, this));

        _loggingControlMsg = std::make_shared<ImageLoggingControlMessage>();
        AddPublishSubscribeMessage("ImageLoggingControlMessage", _loggingControlMsg);

        _brakePositionControlOutpMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        AddPublishSubscribeMessage("BrakePositionControlOutputMessage", _brakePositionControlOutpMsg);

        _throttlePositionControlOutpMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        AddPublishSubscribeMessage("ThrottlePositionControlOutputMessage", _throttlePositionControlOutpMsg);


        //Wake-up manager if any of these messages change
        //_brakeActuator.LinearActuatorPositionControlMsg->Register_SomethingPublished(boost::bind(&VehicleActuatorInterfaceManager::WakeUpManagerEH, this));
        //_throttleActuator.LinearActuatorPositionControlMsg->Register_SomethingPublished(boost::bind(&VehicleActuatorInterfaceManager::WakeUpManagerEH, this));
        _steeringControl.SteeringTorqueCtrlMsg->Register_SomethingPublished(boost::bind(&VehicleActuatorInterfaceManager::WakeUpManagerEH, this));

        _canRxDataRecorder.setDirectory(DataLogDirectory);
        _canTxDataRecorder.setDirectory(DataLogDirectory);

        _brakePosCtrlRecord.RecordType = DataRecordType_e::KTLA_Brake_Cmd;
        _brakePosCtrlRecord.PositionControlMsg = _brakePositionControlOutpMsg;

        _throttlePosCtrlRecord.RecordType = DataRecordType_e::KTLA_Throttle_Cmd;
        _throttlePosCtrlRecord.PositionControlMsg = _throttlePositionControlOutpMsg;

        _steeringCommandDataRecord.SteeringTorqueCtrlMsg = _steeringControl.SteeringTorqueCtrlMsg;

        string fn = config->GetConfigStringValue("VehicleActuatorInterface.DataCANRxLogBaseFilename", "VehicleActuatorCANRxDataLog");
        _canRxDataRecorder.setBaseFilename(fn);
        _canRxDataRecorder.setHeaderRecord(&_dataRecorderCanRxHeader);

        _canRxDataLogReader.setBaseFilename(fn);
        string playbackLogDir = config->GetConfigStringValue("GlobalParameters.PlaybackDataLogDirectory", "PlaybackDataLogs");
        _canRxDataLogReader.setDirectoryName(playbackLogDir);
        _canRxDataLogReader.LoopBackToStartOfDataRecords = false;

        fn = config->GetConfigStringValue("VehicleActuatorInterface.DataCANTxLogBaseFilename", "VehicleActuatorCANTxDataLog");
        _canTxDataRecorder.setBaseFilename(fn);
        _canTxDataRecorder.setHeaderRecord(&_dataRecorderCanTxHeader);

        EnableSteeringLogging = config->GetConfigBoolValue("VehicleActuatorInterface.EnableSteeringLogging", true);
        EnableBrakeLogging = config->GetConfigBoolValue("VehicleActuatorInterface.EnableBrakeLogging", true);
        EnableThrottleLogging = config->GetConfigBoolValue("VehicleActuatorInterface.EnableThrottleLogging", true);

        _brakeActuator.CanRxDataRecorderPtr = &_canRxDataRecorder;
        _brakeActuator.LogActuatorPostionFeedbackData = EnableBrakeLogging;

        _throttleActuator.CanRxDataRecorderPtr = &_canRxDataRecorder;
        _throttleActuator.LogActuatorPostionFeedbackData = EnableThrottleLogging;

        _steeringControl.CanRxDataRecorderPtr = &_canRxDataRecorder;
        _steeringControl.EnableLoggingSteeringStatus = EnableSteeringLogging;

        _brakeActuatorStateCntr = 0;
        _throttleActuatorStateCntr = 0;
        _steeringStateCntr = 0;

        _ctrlStopwatch.reset();
    }

    void VehicleActuatorInterfaceManager::Initialize()
    {

        LOGINFO("VehicleActuatorInterfaceManager: Initialization Started");
        this->SetWakeupTimeDelayMSec(19);

        string canBusId = _config_sptr->GetConfigStringValue("VehicleActuatorInterfaceEnabled.CANBusID", "can0");
        _epasSteeringTorqueReportID = (uint32_t)_config_sptr->GetConfigIntValue("VehicleActuatorInterfaceEnabled.EPASCanTorqueRptID", 0x0290);
        _epasSteeringAngleReportID = (uint32_t)_config_sptr->GetConfigIntValue("VehicleActuatorInterfaceEnabled.EPASCanSteeringAngleRptID", 0x0292);
        _epasSteeringControlID = (uint32_t)_config_sptr->GetConfigIntValue("VehicleActuatorInterfaceEnabled.EPASCanCmdID", 0x0296);

        _brakeActuatorStateCntr = 0;
        _throttleActuatorStateCntr = 0;
        _steeringStateCntr = 0;

        try
        {
            //Open the
            _canSocketHandle = socket(PF_CAN, SOCK_RAW, CAN_RAW);
            if (_canSocketHandle >= 0)
            {
                struct ifreq ifr;
                struct sockaddr_can canAddr;
                strcpy(ifr.ifr_name, canBusId.c_str() );
                ioctl(_canSocketHandle, SIOCGIFINDEX, &ifr);

                canAddr.can_family = AF_CAN;
                canAddr.can_ifindex = ifr.ifr_ifindex;

                if( bind(_canSocketHandle, (struct sockaddr *)&canAddr, sizeof(canAddr)) )
                {
                    LOGERROR("VehicleActuatorInterfaceManager: Could not bind to the CAN Socket.");
                    std::cout << "VehicleActuatorInterfaceManager: Could not bind to the  CAN Socket" << std::endl;
                    return;
                }

                //Setup and start the Receive Background Thread
                _shutdown = false;
                _backgroundRxThread = std::thread(&VehicleActuatorInterfaceManager::receiveCanMessagesThread, this);

                //usleep(100000);
                //Set a couple of parameters
                //_brakeActuator.

                LOGINFO("VehicleActuatorInterfaceManager: Initialization Complete");
                std::cout << "VehicleActuatorInterfaceManager: Initialization Complete" << std::endl;
            }
            else
            {
                LOGERROR("VehicleActuatorInterfaceManager: Could not open CAN Socket.");
                std::cout << "VehicleActuatorInterfaceManager: Could not open CAN Socket" << std::endl;
            }

            //Read the Default Data Log Directory to see if there are log files;
            _canRxDataLogReader.GetListOfFilesFromDirectory();
        }
        catch(exception e)
        {
            LOGERROR("VehicleActuatorInterfaceManager: Event open exception: " << e.what());
        }
    }


    void VehicleActuatorInterfaceManager::sendKarTechParametetersToActuator(
            std::shared_ptr<KarTechLinearActuatorParamsMessage> LinearActuatorParamsMsg,
            std::shared_ptr<KarTechLinearActuatorParamsMessage> CurrentLinearActuatorParamsMsg,
            bool forceSendAll)
    {
        std::string ActuatorType = "Unknown";
        KarTechLinearActuator *ktlaPtr = nullptr;
        if(LinearActuatorParamsMsg->FunctionType == LinearActuatorFunction_e::LA_Brake)
        {
            ktlaPtr = &_brakeActuator;
            ActuatorType = "Brake";
        }
        else if(LinearActuatorParamsMsg->FunctionType == LinearActuatorFunction_e::LA_Accelerator)
        {
            ktlaPtr = &_throttleActuator;
            ActuatorType = "Throttle";
        }
        else
            return;   //Not a valid actuator.

        //Force the Min/Max Positions in the Feedback Msg to be the new set values...
        //this is the only way they are set.
        ktlaPtr->setMinActuatorPositionInches(LinearActuatorParamsMsg->getMinPositionInches());
        ktlaPtr->setMaxActuatorPositionInches(LinearActuatorParamsMsg->getMaxPositionInches());
        ktlaPtr->LinearActuatorParamsFeedbackMsg->setMinPositionInches(ktlaPtr->getMinActuatorPositionInches());
        ktlaPtr->LinearActuatorParamsFeedbackMsg->setMaxPositionInches(ktlaPtr->getMaxActuatorPositionInches());
        ktlaPtr->LinearActuatorParamsFeedbackMsg->PostMessage();
        LOGINFO("KarTeck LA Set [" << ActuatorType << "] MinPositionInches: " << LinearActuatorParamsMsg->getMinPositionInches()
                                   << " MaxPositionInches: " << LinearActuatorParamsMsg->getMaxPositionInches());

        if(forceSendAll || LinearActuatorParamsMsg->getMotorMaxCurrentLimitAmps() != CurrentLinearActuatorParamsMsg->getMotorMaxCurrentLimitAmps())
        {
            ktlaPtr->generateMotorOverCurrentConfigMsg(_txCanFame , LinearActuatorParamsMsg->getMotorMaxCurrentLimitAmps());
            sendCanMessage(_txCanFame);
            LOGINFO("KarTeck LA Set [" << ActuatorType << "] MotorMaxCurrentLimit: " << LinearActuatorParamsMsg->getMotorMaxCurrentLimitAmps());
            usleep(2500);
        }

        if(forceSendAll || LinearActuatorParamsMsg->getPositionReachedErrorTimeMSec() != CurrentLinearActuatorParamsMsg->getPositionReachedErrorTimeMSec())
        {
            ktlaPtr->generatePositionReachedErrorTimeMsg(_txCanFame , LinearActuatorParamsMsg->getPositionReachedErrorTimeMSec());
            sendCanMessage(_txCanFame);
            LOGINFO("KarTeck LA Set [" << ActuatorType << "] PositionReachedErrorTime: " << LinearActuatorParamsMsg->getPositionReachedErrorTimeMSec());
            usleep(2500);
        }

        if(forceSendAll || LinearActuatorParamsMsg->getFeedbackCtrl_KP() != CurrentLinearActuatorParamsMsg->getFeedbackCtrl_KP()
                || LinearActuatorParamsMsg->getFeedbackCtrl_KI() != CurrentLinearActuatorParamsMsg->getFeedbackCtrl_KI())
        {
            ktlaPtr->generateConfigure_Kp_Ki_Msg(_txCanFame ,
                                                 LinearActuatorParamsMsg->getFeedbackCtrl_KP(),
                                                 LinearActuatorParamsMsg->getFeedbackCtrl_KI());
            sendCanMessage(_txCanFame);
            LOGINFO("KarTeck LA Set [" << ActuatorType << "] FeedbackCtrl_KP: " << LinearActuatorParamsMsg->getFeedbackCtrl_KP()
                 << " FeedbackCtrl_KI: " << LinearActuatorParamsMsg->getFeedbackCtrl_KI());
            usleep(2500);
        }

        if(forceSendAll || LinearActuatorParamsMsg->getFeedbackCtrl_KD() != CurrentLinearActuatorParamsMsg->getFeedbackCtrl_KD()
           || LinearActuatorParamsMsg->getFeedbackCtrl_CLFreq() != CurrentLinearActuatorParamsMsg->getFeedbackCtrl_CLFreq()
                || LinearActuatorParamsMsg->getFeedbackCtrl_ErrDeadbandInces() != CurrentLinearActuatorParamsMsg->getFeedbackCtrl_ErrDeadbandInces())
        {
            ktlaPtr->generateConfigure_KD_Freq_EDB_Msg(_txCanFame ,
                                                 LinearActuatorParamsMsg->getFeedbackCtrl_KD(),
                                                 LinearActuatorParamsMsg->getFeedbackCtrl_CLFreq(),
                                                       LinearActuatorParamsMsg->getFeedbackCtrl_ErrDeadbandInces());
            sendCanMessage(_txCanFame);
            LOGINFO("KarTeck LA Set [" << ActuatorType << "] FeedbackCtrl_KD: " << LinearActuatorParamsMsg->getFeedbackCtrl_KD()
                                       << " FeedbackCtrl_CLFreq: " << LinearActuatorParamsMsg->getFeedbackCtrl_CLFreq()
                                       << " ErrDeadbandInces: " << LinearActuatorParamsMsg->getFeedbackCtrl_ErrDeadbandInces());
            usleep(2500);
        }

        if(forceSendAll || LinearActuatorParamsMsg->getMotor_MinPWM() != CurrentLinearActuatorParamsMsg->getMotor_MinPWM()
           || LinearActuatorParamsMsg->getMotor_MaxPWM() != CurrentLinearActuatorParamsMsg->getMotor_MaxPWM()
           || LinearActuatorParamsMsg->getMotor_pwmFreq() != CurrentLinearActuatorParamsMsg->getMotor_pwmFreq())
        {
            ktlaPtr->generateConfigureMotorPWM_Msg(_txCanFame ,
                                                       LinearActuatorParamsMsg->getMotor_MinPWM(),
                                                       LinearActuatorParamsMsg->getMotor_MaxPWM(),
                                                       LinearActuatorParamsMsg->getMotor_pwmFreq());
            sendCanMessage(_txCanFame);
            LOGINFO("KarTeck LA Set [" << ActuatorType << "] Motor_MinPWM: " << LinearActuatorParamsMsg->getMotor_MinPWM()
                                       << " Motor_MaxPWM: " << LinearActuatorParamsMsg->getMotor_MaxPWM()
                                       << " Motor_pwmFreq: " << LinearActuatorParamsMsg->getMotor_pwmFreq());
            usleep(2500);
        }

    }

    //The linear actuators work best if the clutch is engaged before the motor
    //and vice-versa when disabling... this code controls that process.
    //It assumes this method is called on a regular basis.
    void VehicleActuatorInterfaceManager::setBrakeActuatorPostion()
    {
        std::shared_ptr<LinearActuatorPositionCtrlMessage> pcMsgPtr;
        bool clutchEnable = true;
        bool motorEnable = true;
        double actPosInches = 0;
         _brakeActuator.PositionControlMsg->FetchMessage();
        pcMsgPtr = _brakeActuator.PositionControlMsg;
        _brakeActuator.LinearActuatorPositionFeedbackMsg->ManualExtControl = _brakeActuator.PositionControlMsg->ManualExtControl;
        _brakeActuator.LinearActuatorPositionFeedbackMsg->ActuatorSetupMode = _brakeActuator.PositionControlMsg->ActuatorSetupMode;
        actPosInches = _brakeActuator.postionPercentToInches(pcMsgPtr->getPositionPercent());
        _brakePosCtrlRecord.PositionSetInches = actPosInches;
        if(pcMsgPtr->ActuatorSetupMode)
        {
            //We are in a special setup mode...
            if(_brakeActuator.LinearActuatorSetupMsg->FetchMessage())
            {
                //Only one type of Setup can be done at a time.
                if(_brakeActuator.LinearActuatorSetupMsg->ResetOutputs)
                {
                    _brakeActuator.generateResetActuatorMsg(_txBrakeCanFame, 0x0001, 0x0000);
                    sendCanMessage(_txBrakeCanFame);
                    LOGINFO("KarTeck LA Brake Actuator: Reset Outputs" );

                }
                else if(_brakeActuator.LinearActuatorSetupMsg->ResetHardwareCfgs)
                {
                    _brakeActuator.generateResetActuatorMsg(_txBrakeCanFame, 0x0008, 0x0000);
                    sendCanMessage(_txBrakeCanFame);
                    LOGINFO("KarTeck LA Brake Actuator: Reset Hardware Configs" );
                }
                else if(_brakeActuator.LinearActuatorSetupMsg->ResetUserCfgs)
                {
                    _brakeActuator.generateResetActuatorMsg(_txBrakeCanFame, 0x0010, 0x0000);
                    sendCanMessage(_txBrakeCanFame);
                    LOGINFO("KarTeck LA Brake Actuator: Reset User Configs" );
                }
                else if(_brakeActuator.LinearActuatorSetupMsg->ResetAll)
                {
                    _brakeActuator.generateResetActuatorMsg(_txBrakeCanFame, 0xFFFF, 0x001F);
                    sendCanMessage(_txBrakeCanFame);
                    LOGINFO("KarTeck LA Brake Actuator: Reset All" );
                }
                else if(_brakeActuator.LinearActuatorSetupMsg->AutoZeroCal)
                {
                    _brakeActuator.generateAutoZeroCalibrationMsg(_txBrakeCanFame);
                    sendCanMessage(_txBrakeCanFame);
                    LOGINFO("KarTeck LA Brake Actuator: Auto Zero Calibration" );
                }
                else if(_brakeActuator.LinearActuatorSetupMsg->SetCanCommandResponsIDs)
                {
                    _brakeActuator.generateSetCommandIDMsg(_txBrakeCanFame, _brakeActuator.BrakeCmdID);
                    sendCanMessage(_txBrakeCanFame);
                    LOGINFO("KarTeck LA Brake Actuator: Set Command ID: " << _brakeActuator.BrakeCmdID );
                    usleep(2500);
                    _brakeActuator.generateSetReportIDMsg(_txBrakeCanFame, true, _brakeActuator.BrakeRptID, true);
                    sendCanMessage(_txBrakeCanFame);
                    LOGINFO("KarTeck LA Brake Actuator: Set Report ID: " << _brakeActuator.BrakeRptID);
                    usleep(2500);
                }

                //Force clearing of the setup message to ensure the setup
                //only occurs once and there is no lingering setup.
                _brakeActuator.LinearActuatorSetupMsg->Clear();
                _brakeActuator.LinearActuatorSetupMsg->PostMessage();
            }

            if( _brakeActuator.LinearActuatorParamsControlMsg->FetchMessage())
            {
                //Force the Actuator type incase Sender does not set the actuator type
                _brakeActuator.LinearActuatorParamsControlMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
                sendKarTechParametetersToActuator(_brakeActuator.LinearActuatorParamsControlMsg,
                                                  _brakeActuator.LinearActuatorParamsFeedbackMsg,
                                                   true);
            }

        }
        else if(pcMsgPtr->MotorEnable == false || pcMsgPtr->ClutchEnable == false)
        {
            //Disable the Motor then the Clutch
            if(_brakeActuator.LinearActuatorPositionFeedbackMsg->MotorEnable)
            {
                motorEnable = false;
                clutchEnable = _brakeActuator.LinearActuatorPositionFeedbackMsg->ClutchEnable;
                //Disable the motor on this round
                _brakeActuator.generateActuatorPositionInchesMsg(_txBrakeCanFame,
                                                                 actPosInches,
                                                                 clutchEnable, motorEnable);
                sendCanMessage(_txBrakeCanFame);
            }
            else if(_brakeActuator.LinearActuatorPositionFeedbackMsg->ClutchEnable)
            {
                motorEnable = false;
                clutchEnable = false;
                _brakeActuator.generateActuatorPositionInchesMsg(_txBrakeCanFame,
                                                                 actPosInches,
                                                                 clutchEnable, motorEnable);
                sendCanMessage(_txBrakeCanFame);
            }

            _brakeActuatorStateCntr = 0;

            //else, the motor and clutch are disabled... so no need to send motor position
            //commands to the actuator.
        }
        else
        {
            //Enable the KarTech Actuator...
            switch(_brakeActuatorStateCntr)
            {
                case 0:
                    //Ensure the KarTech Actuator has all the correct parameters
                    //as the power to the actuator could have been turned off, so they need to be
                    //reset each time the Actuator is enable.
                    _brakeActuator.LinearActuatorParamsControlMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
                    sendKarTechParametetersToActuator(_brakeActuator.LinearActuatorParamsControlMsg,
                                                      _brakeActuator.LinearActuatorParamsFeedbackMsg, true);
                    ++_brakeActuatorStateCntr;
                    break;

                case 1:
                    _brakeActuator.generateRequestPositionAndMotorCurrentReportMsg(_txBrakeCanFame,
                                                                                   _brakeActuator.getActuatorReportRateMSec());
                    sendCanMessage(_txBrakeCanFame);
                    ++_brakeActuatorStateCntr;
                    break;

                case 2:
                    //enable the clutch before the motor.
                    motorEnable = false;
                    clutchEnable = true;
                    //changing the feedback message here helps ensure the next time
                    // around the loop we will enable the motor.
                    _brakeActuator.generateActuatorPositionInchesMsg(_txBrakeCanFame,
                                                                     actPosInches,
                                                                     clutchEnable, motorEnable);
                    sendCanMessage(_txBrakeCanFame);
                    ++_brakeActuatorStateCntr;
                    break;

                default:
                    //enable the clutch before the motor.
                    motorEnable = true;
                    clutchEnable = true;
                    //changing the feedback message here helps ensure the next time
                    // around the loop we will enable the motor.
                    _brakeActuator.generateActuatorPositionInchesMsg(_txBrakeCanFame,
                                                                     actPosInches,
                                                                     clutchEnable, motorEnable);
                    sendCanMessage(_txBrakeCanFame);
                    break;
            }

        }

        _brakePositionControlOutpMsg->setPositionPercent(pcMsgPtr->getPositionPercent());
        _brakePositionControlOutpMsg->ClutchEnable = clutchEnable;
        _brakePositionControlOutpMsg->MotorEnable = motorEnable;
        _brakePositionControlOutpMsg->ManualExtControl = pcMsgPtr->ManualExtControl;
        _brakePositionControlOutpMsg->ActuatorSetupMode = pcMsgPtr->ActuatorSetupMode;
        _brakePositionControlOutpMsg->PostMessage();

        _brakeActuator.LinearActuatorPositionFeedbackMsg->PostMessage();

        _loggingControlMsg->FetchMessage();
        if(EnableBrakeLogging && _loggingControlMsg->EnableLogging)
        {
            _canTxDataRecorder.writeDataRecord(_brakePosCtrlRecord);
        }
        else if(!_loggingControlMsg->EnableLogging )
        {
            _canTxDataRecorder.closeLogFile();
        }
    }


    void VehicleActuatorInterfaceManager::setThrottleActuatorPostion()
    {
        std::shared_ptr<LinearActuatorPositionCtrlMessage> pcMsgPtr;
        bool clutchEnable = true;
        bool motorEnable = true;
        double actPosInches = 0;
        _throttleActuator.PositionControlMsg->FetchMessage();
        pcMsgPtr = _throttleActuator.PositionControlMsg;
        _throttleActuator.LinearActuatorPositionFeedbackMsg->ManualExtControl = pcMsgPtr->ManualExtControl;
        _throttleActuator.LinearActuatorPositionFeedbackMsg->ActuatorSetupMode = pcMsgPtr->ActuatorSetupMode;
        actPosInches = _throttleActuator.postionPercentToInches(pcMsgPtr->getPositionPercent());
        _throttlePosCtrlRecord.PositionSetInches = actPosInches;
        if(pcMsgPtr->ActuatorSetupMode)
        {
            //We are in a special setup mode...
            if(_throttleActuator.LinearActuatorSetupMsg->FetchMessage())
            {
                //Only one type of Setup can be done at a time.
                if(_throttleActuator.LinearActuatorSetupMsg->ResetOutputs)
                {
                    _throttleActuator.generateResetActuatorMsg(_txThrottleCanFame, 0x0001, 0x0000);
                    sendCanMessage(_txThrottleCanFame);
                    LOGINFO("KarTeck LA Throttle Actuator: Reset Outputs" );
                }
                else if(_throttleActuator.LinearActuatorSetupMsg->ResetHardwareCfgs)
                {
                    _throttleActuator.generateResetActuatorMsg(_txThrottleCanFame, 0x0008, 0x0000);
                    sendCanMessage(_txThrottleCanFame);
                    LOGINFO("KarTeck LA Throttle Actuator: Reset Hardware Configs" );
                }
                else if(_throttleActuator.LinearActuatorSetupMsg->ResetUserCfgs)
                {
                    _throttleActuator.generateResetActuatorMsg(_txThrottleCanFame, 0x0010, 0x0000);
                    sendCanMessage(_txThrottleCanFame);
                    LOGINFO("KarTeck LA Throttle Actuator: Reset User Configs" );
                }
                else if(_throttleActuator.LinearActuatorSetupMsg->ResetAll)
                {
                    _throttleActuator.generateResetActuatorMsg(_txThrottleCanFame, 0xFFFF, 0x001F);
                    sendCanMessage(_txThrottleCanFame);
                    LOGINFO("KarTeck LA Throttle Actuator: Reset All" );
                }
                else if(_throttleActuator.LinearActuatorSetupMsg->AutoZeroCal)
                {
                    _throttleActuator.generateAutoZeroCalibrationMsg(_txThrottleCanFame);
                    sendCanMessage(_txThrottleCanFame);
                    LOGINFO("KarTeck LA Throttle Actuator: Auto Zero Calibration" );
                }
                else if(_throttleActuator.LinearActuatorSetupMsg->SetCanCommandResponsIDs)
                {
                    _throttleActuator.generateSetCommandIDMsg(_txThrottleCanFame, _throttleActuator.AcceleratorCmdID);
                    sendCanMessage(_txThrottleCanFame);
                    LOGINFO("KarTeck LA Throttle Actuator: Set Command ID: " << _throttleActuator.AcceleratorCmdID)
                    usleep(2500);
                    _throttleActuator.generateSetReportIDMsg(_txThrottleCanFame, true, _throttleActuator.AcceleratorRptID, true);
                    sendCanMessage(_txThrottleCanFame);
                    LOGINFO("KarTeck LA Throttle Actuator: Set Report ID: " << _throttleActuator.AcceleratorRptID);
                    usleep(2500);
                }

                //Force clearing of the setup message to ensure the setup
                //only occurs once and there is no lingering setup.
                _throttleActuator.LinearActuatorSetupMsg->Clear();
                _throttleActuator.LinearActuatorSetupMsg->PostMessage();
            }

            if( _throttleActuator.LinearActuatorParamsControlMsg->FetchMessage())
            {
                //Force the Actuator type incase Sender does not set the actuator type
                _throttleActuator.LinearActuatorParamsControlMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
                sendKarTechParametetersToActuator(_throttleActuator.LinearActuatorParamsControlMsg,
                                                  _throttleActuator.LinearActuatorParamsFeedbackMsg,
                                                  true);
            }

        }
        else if(pcMsgPtr->MotorEnable == false || pcMsgPtr->ClutchEnable == false)
        {
            //Disable the Motor then the Clutch
            if(_throttleActuator.LinearActuatorPositionFeedbackMsg->MotorEnable)
            {
                motorEnable = false;
                clutchEnable = _throttleActuator.LinearActuatorPositionFeedbackMsg->ClutchEnable;
                //Disable the motor on this round
                _throttleActuator.generateActuatorPositionInchesMsg(_txThrottleCanFame,
                                                                 actPosInches,
                                                                 clutchEnable, motorEnable);
                sendCanMessage(_txThrottleCanFame);
            }
            else if(_throttleActuator.LinearActuatorPositionFeedbackMsg->ClutchEnable)
            {
                motorEnable = false;
                clutchEnable = false;
                _throttleActuator.generateActuatorPositionInchesMsg(_txThrottleCanFame,
                                                                 actPosInches,
                                                                 clutchEnable, motorEnable);
                sendCanMessage(_txThrottleCanFame);
            }

            _throttleActuatorStateCntr = 0;
            //else, the motor and clutch are disabled... so no need to send motor position
            //commands to the actuator.
        }
        else
        {
            //Enable the KarTech Actuator...
            switch(_throttleActuatorStateCntr)
            {
                case 0:
                    //Ensure the KarTech Actuator has all the correct parameters
                    //as the power to the actuator could have been turned off, so they need to be
                    //reset each time the Actuator is enable.
                    _throttleActuator.LinearActuatorParamsControlMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
                    sendKarTechParametetersToActuator(_throttleActuator.LinearActuatorParamsControlMsg,
                                                      _throttleActuator.LinearActuatorParamsFeedbackMsg, true);
                    ++_throttleActuatorStateCntr;
                    break;

                case 1:
                    _throttleActuator.generateRequestPositionAndMotorCurrentReportMsg(_txThrottleCanFame,
                                                                                      _throttleActuator.getActuatorReportRateMSec());
                    sendCanMessage(_txThrottleCanFame);
                    ++_throttleActuatorStateCntr;
                    break;

                case 2:
                    //enable the clutch before the motor.
                    motorEnable = false;
                    clutchEnable = true;
                    //changing the feedback message here helps ensure the next time
                    // around the loop we will enable the motor.
                    _throttleActuator.generateActuatorPositionInchesMsg(_txThrottleCanFame,
                                                                     actPosInches,
                                                                     clutchEnable, motorEnable);
                    sendCanMessage(_txThrottleCanFame);
                    ++_throttleActuatorStateCntr;
                    break;

                default:
                    //enable the clutch before the motor.
                    motorEnable = true;
                    clutchEnable = true;
                    //changing the feedback message here helps ensure the next time
                    // around the loop we will enable the motor.
                    _throttleActuator.generateActuatorPositionInchesMsg(_txThrottleCanFame,
                                                                     actPosInches,
                                                                     clutchEnable, motorEnable);
                    sendCanMessage(_txThrottleCanFame);
                    break;
            }

        }

        _throttlePositionControlOutpMsg->setPositionPercent(pcMsgPtr->getPositionPercent());
        _throttlePositionControlOutpMsg->ClutchEnable = clutchEnable;
        _throttlePositionControlOutpMsg->MotorEnable = motorEnable;
        _throttlePositionControlOutpMsg->ManualExtControl = pcMsgPtr->ManualExtControl;
        _throttlePositionControlOutpMsg->ActuatorSetupMode = pcMsgPtr->ActuatorSetupMode;;
        _throttlePositionControlOutpMsg->PostMessage();

        _throttleActuator.LinearActuatorPositionFeedbackMsg->PostMessage();

        _loggingControlMsg->FetchMessage();
        if(EnableThrottleLogging && _loggingControlMsg->EnableLogging)
        {
            _canTxDataRecorder.writeDataRecord(_throttlePosCtrlRecord);
        }
        else if(!_loggingControlMsg->EnableLogging )
        {
            _canTxDataRecorder.closeLogFile();
        }

    }


    void VehicleActuatorInterfaceManager::setSteeringControlTorque()
    {
        std::shared_ptr<SteeringTorqueCtrlMessage> scMsgPtr;
        _steeringControl.SteeringTorqueCtrlMsg->FetchMessage();
        scMsgPtr = _steeringControl.SteeringTorqueCtrlMsg;
        double steeringTorquePercent = scMsgPtr->getSteeringTorquePercent();
        _steeringControl.setSteeringTorqueMap(scMsgPtr->getSteeringTorqueMap());
        _steeringControl.generateSteeringTorqueControlMsg(_txSteeringCanFame, steeringTorquePercent);
        sendCanMessage(_txSteeringCanFame);

        _loggingControlMsg->FetchMessage();
        if(EnableSteeringLogging && _loggingControlMsg->EnableLogging)
        {
            _steeringCommandDataRecord.SteeringTorqueCtrlMsg = scMsgPtr;
            _canTxDataRecorder.writeDataRecord(_steeringCommandDataRecord);
        }
        else if(!_loggingControlMsg->EnableLogging )
        {
            _canTxDataRecorder.closeLogFile();
        }
    }

    void VehicleActuatorInterfaceManager::testActuatorControl()
    {
        static double periodTsec = 5.0;
        _ctrlStopwatch.captureTime();
        double tsec = _ctrlStopwatch.getTimeElapsed();
        if(tsec > 0)
        {
            tsec = std::fmod(tsec, periodTsec);
            double angleRad = 2.0 * boost::math::constants::pi<double>() * tsec / periodTsec;
            double pp = 100.0 * std::sin(angleRad);
            if(pp > 0)
            {
                _throttleActuator.PositionControlMsg->setPositionPercent(pp, true);
                _brakeActuator.PositionControlMsg->setPositionPercent(0.0, true);
            }
            else
            {
                _throttleActuator.PositionControlMsg->setPositionPercent(0.0, true);
                _brakeActuator.PositionControlMsg->setPositionPercent(-pp, true);
            }
            _throttleActuator.PositionControlMsg->PostMessage();
            _brakeActuator.PositionControlMsg->PostMessage();
        }
        else
        {
            _ctrlStopwatch.reset();
            _ctrlStopwatch.start();
            _throttleActuator.PositionControlMsg->setPositionPercent(0.0, true);
            _brakeActuator.PositionControlMsg->setPositionPercent(0.0, true);
            _throttleActuator.PositionControlMsg->PostMessage();
            _brakeActuator.PositionControlMsg->PostMessage();
        }
    }


    void VehicleActuatorInterfaceManager::Startup()
    {
        _brakeActuator.generateRequestPositionAndMotorCurrentReportMsg(_txCanFame, _brakeActuator.getActuatorReportRateMSec());
        sendCanMessage(_txCanFame);
        usleep(10000);
        sendKarTechParametetersToActuator(_brakeActuator.LinearActuatorParamsControlMsg,
                                          _brakeActuator.LinearActuatorParamsFeedbackMsg, true);
        usleep(200000);

        //Send message to the actuator to report position and current
        _throttleActuator.generateRequestPositionAndMotorCurrentReportMsg(_txCanFame, _throttleActuator.getActuatorReportRateMSec());
        sendCanMessage(_txCanFame);
        usleep(10000);
        sendKarTechParametetersToActuator(_throttleActuator.LinearActuatorParamsControlMsg,
                                          _throttleActuator.LinearActuatorParamsFeedbackMsg, true);
        usleep(200000);
    }

    void VehicleActuatorInterfaceManager::ExecuteUnitOfWork()
    {
        static double steeringCtrlTorque = 0;

        //testActuatorControl();

        setBrakeActuatorPostion();
        setThrottleActuatorPostion();
        setSteeringControlTorque();


        /*****************************************************************
        if(_msgCntr % 400 == 0)
        {
            //Send message to the actuator to report position and current
            _brakeActuator.generateRequestPositionAndMotorCurrentReportMsg(_txCanFame, _brakeActuator.getActuatorReportRateMSec());
            sendCanMessage(_txCanFame);
        }
        if(_msgCntr % 400 == 1)
        {
            //Send message to the actuator to report position and current
            _throttleActuator.generateRequestPositionAndMotorCurrentReportMsg(_txCanFame, _throttleActuator.getActuatorReportRateMSec());
            sendCanMessage(_txCanFame);
        }
         **********************************************************************/

        /************Test Log Reader ****************************
        std::shared_ptr<DataRecorderAbstractRecord> recordPtr;
        recordPtr = _canRxDataLogReader.ReadNextRecord();
        if(recordPtr.get() != nullptr)
        {
            //std::cout << "Record Read Type: " << recordPtr->RecordType
            //          << " Timestamp: " << recordPtr->TimeStampSec << std::endl;
        }
        else
        {
            std::cout << "Could not read record!" << std::endl;
        }
         ******************************************************/

        ++_msgCntr;
    }


    void VehicleActuatorInterfaceManager::Shutdown()
    {
        _shutdown = true;
        _canTxDataRecorder.closeLogFile();
        usleep(250000);
        if(_canSocketHandle >= 0)
        {
            close(_canSocketHandle);
        }
        _backgroundRxThread.join();
        //Don't close the Rx Logger until the thread shuts down.
        _canRxDataRecorder.closeLogFile();
        LOGINFO("VehicleActuatorInterfaceManager shutdown");
    }

    bool VehicleActuatorInterfaceManager::sendCanMessage(struct can_frame &txCanMsg)
    {
        if( txCanMsg.can_id > CAN_SFF_MASK)
        {
            //Assume this is an extended CAN Fram
            txCanMsg.can_id |= CAN_EFF_FLAG;
        }
        try
        {
            int nbytes = write(_canSocketHandle, &txCanMsg, sizeof(struct can_frame));
            //usleep(1000);
            if(nbytes != sizeof(struct can_frame) )
            {
                //LOGERROR("VehicleActuatorInterfaceManager: CAN Socet Tx Send error, nbytes: " << nbytes);
            }
        }
        catch(exception e)
        {
            LOGERROR("VehicleActuatorInterfaceManager: CAN Socet Tx Exception: " << e.what());
        }
    }

    void VehicleActuatorInterfaceManager::receiveCanMessagesThread()
    {
        _backgroundRxThreadIsRunning = true;
        while(!_shutdown)
        {
            try
            {
                //Blocking Read for a CAN Frame
                int nbytes = read(_canSocketHandle, &_rxCanFame, sizeof(struct can_frame));
                if(nbytes >= sizeof(struct can_frame))
                {
                    //Clear the upper flag bits
                    _rxCanFame.can_id &= CAN_EFF_MASK;
                    if(_rxCanFame.can_id == _steeringControl.SteeringTorqueRptID)
                    {
                        _steeringControl.processSteeringTorqueMsg(_rxCanFame);
                    }
                    else if(_rxCanFame.can_id == _steeringControl.SteeringAngleRptID)
                    {
                        _steeringControl.processSteeringAngleMsg(_rxCanFame);
                    }
                    else if(_rxCanFame.can_id == _brakeActuator.getCanRptID())
                    {
                        _brakeActuator.processRxCanMessage(_rxCanFame);
                    }
                    else if(_rxCanFame.can_id == _throttleActuator.getCanRptID())
                    {
                        _throttleActuator.processRxCanMessage(_rxCanFame);
                    }
                    else
                    {
                        LOGWARN("Received Unknown Can Msg ID: " << _rxCanFame.can_id);
                    }
                }
                else
                {
                    LOGERROR("VehicleActuatorInterfaceManager: Invalid CAN message received.");
                }
            }
            catch(exception e)
            {
                LOGERROR("VehicleActuatorInterfaceManager: CAN Socet Rx Exception: " << e.what());
            }

        }
        _backgroundRxThreadIsRunning = false;
    }

}

