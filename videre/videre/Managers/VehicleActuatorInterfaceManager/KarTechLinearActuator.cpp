/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: June 1, 2018
 *
 * KarTech Linear Actuator Interface
 *******************************************************************/


#include "KarTechLinearActuator.h"

using namespace std;

namespace videre
{

    KarTechLinearActuator::KarTechLinearActuator(Rabit::RabitManager *mgrPtr,
                                                 LinearActuatorFunction_e functionType,
                                                 std::shared_ptr<ConfigData> config)
        : KTLAStatusDataRecord()
    {
        _mgrPtr = mgrPtr;
        _config = config;
        _functionType = functionType;
        _pushPullType = GetLinearActuatorPullPushType(functionType);
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        PositionControlMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        LinearActuatorPositionFeedbackMsg = std::make_shared<LinearActuatorPositionCtrlMessage>();
        LinearActuatorParamsControlMsg = std::make_shared<KarTechLinearActuatorParamsMessage>();
        LinearActuatorParamsFeedbackMsg = std::make_shared<KarTechLinearActuatorParamsMessage>();
        LinearActuatorSetupMsg = std::make_shared<KarTechLinearActuatorSetupMessage>();

        _loggingControlMsg = std::make_shared<ImageLoggingControlMessage>();
        _mgrPtr->AddPublishSubscribeMessage("ImageLoggingControlMessage", _loggingControlMsg);

        //_mgrPtr->WakeUpManagerOnMessagePost(LinearActuatorPositionControlMsg);
        //_mgrPtr->WakeUpManagerOnMessagePost(LinearActuatorParamsControlMsg);

        if(functionType == LinearActuatorFunction_e::LA_Brake)
        {
            _commandID = (uint32_t) _config->GetConfigIntValue("VehicleActuatorInterfaceEnabled.BrakeCanCmdID",
                                                               BrakeCmdID);
            _reportID = (uint32_t) _config->GetConfigIntValue("VehicleActuatorInterfaceEnabled.BrakeCanRptID",
                                                              BrakeRptID);
            _mgrPtr->AddPublishSubscribeMessage("BrakeLAPositionControlMsg", PositionControlMsg);
            _mgrPtr->AddPublishSubscribeMessage("BrakeLAPositionFeedbackMsg", LinearActuatorPositionFeedbackMsg);
            _mgrPtr->AddPublishSubscribeMessage("BrakeLAParamsControlMsg", LinearActuatorParamsControlMsg);
            _mgrPtr->AddPublishSubscribeMessage("BrakeLAParamsFeedbackMsg", LinearActuatorParamsFeedbackMsg);
            _mgrPtr->AddPublishSubscribeMessage("BrakeLASetupMsg", LinearActuatorSetupMsg);

            PositionControlMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
            LinearActuatorPositionFeedbackMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
            LinearActuatorParamsControlMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
            LinearActuatorParamsFeedbackMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;
            LinearActuatorSetupMsg->FunctionType = LinearActuatorFunction_e::LA_Brake;

            KTLAStatusDataRecord.RecordType = DataRecordType_e::KTLA_Brake_Status;
            KTLAStatusDataRecord.LinearActuatorStatusFeedbackMsg = LinearActuatorPositionFeedbackMsg;
        }
        else if(functionType == LinearActuatorFunction_e::LA_Accelerator)
        {
            _commandID = (uint32_t) _config->GetConfigIntValue("VehicleActuatorInterfaceEnabled.AcceleratorCanCmdID",
                                                               AcceleratorCmdID);
            _reportID = (uint32_t) _config->GetConfigIntValue("VehicleActuatorInterfaceEnabled.AcceleratorCanRptID",
                                                              AcceleratorRptID);
            _mgrPtr->AddPublishSubscribeMessage("ThrottleLAPositionControlMsg", PositionControlMsg);
            _mgrPtr->AddPublishSubscribeMessage("ThrottleLAPositionFeedbackMsg", LinearActuatorPositionFeedbackMsg);
            _mgrPtr->AddPublishSubscribeMessage("ThrottleLAParamsControlMsg", LinearActuatorParamsControlMsg);
            _mgrPtr->AddPublishSubscribeMessage("ThrottleLAParamsFeedbackMsg", LinearActuatorParamsFeedbackMsg);
            _mgrPtr->AddPublishSubscribeMessage("ThrottleLASetupMsg", LinearActuatorSetupMsg);

            PositionControlMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
            LinearActuatorPositionFeedbackMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
            LinearActuatorParamsControlMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
            LinearActuatorParamsFeedbackMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;
            LinearActuatorSetupMsg->FunctionType = LinearActuatorFunction_e::LA_Accelerator;

            KTLAStatusDataRecord.RecordType = DataRecordType_e::KTLA_Throttle_Status;
            KTLAStatusDataRecord.LinearActuatorStatusFeedbackMsg = LinearActuatorPositionFeedbackMsg;
        }

        readParametersConfig(LinearActuatorParamsControlMsg);
        LinearActuatorParamsControlMsg->PostMessage();
    }


    double KarTechLinearActuator::postionPercentToInches(double valPercent)
    {
        double posInches = 0;
        valPercent = valPercent < 0 ? 0 : valPercent > 100.0 ? 100.0 : valPercent;
        if(_pushPullType == LinearActuatorPullPushType_e::LAPP_Push)
        {
            posInches = _minActuatorPositionInches + 0.01 * valPercent * getActuatorRangeInches();
        }
        else  //pull type
        {
            posInches = _maxActuatorPositionInches - 0.01 * valPercent * getActuatorRangeInches();
        }
        return posInches;
    }

    //This will may return values outside of the range of [0, 100.0]
    //since the acuatal position can be outside the min max values.
    double KarTechLinearActuator::postionInchesToPercent(double posInches)
    {
        double posPercent = 0;
        posInches = posInches < MinActuatorAbsolutePositionInches ? MinActuatorAbsolutePositionInches
                : posInches > MaxActuatorAbsolutePositionInches ? MaxActuatorAbsolutePositionInches : posInches;
        if(_pushPullType == LinearActuatorPullPushType_e::LAPP_Push)
        {
            posInches = posInches - _minActuatorPositionInches;
        }
        else  //pull type
        {
            posInches = _maxActuatorPositionInches - posInches;
        }
        posPercent = 100.0 * posInches / getActuatorRangeInches();
        return posPercent;
    }

    //Read the KarTech Config Parameters from the config file and store them in the
    //LinearActuatorParamsMsg.
    void KarTechLinearActuator::readParametersConfig(std::shared_ptr<KarTechLinearActuatorParamsMessage> LinearActuatorParamsMsg)
    {
        string base = LinearActuatorParamsMsg->FunctionType == LinearActuatorFunction_e::LA_Brake ?
                "BrkeActuatorConfig." : "ThrottleActuatorConfig.";

        string paramStr = base + "MinPositionInches";
        LinearActuatorParamsMsg->setMinPositionInches(_config->GetConfigDoubleValue(paramStr, 0.0));
        setMinActuatorPositionInches(LinearActuatorParamsMsg->getMinPositionInches());

        paramStr = base + "MaxPositionInches";
        LinearActuatorParamsMsg->setMaxPositionInches(_config->GetConfigDoubleValue(paramStr, 3.0));
        setMaxActuatorPositionInches(LinearActuatorParamsMsg->getMaxPositionInches());

        paramStr = base + "ActuatorReportRateMSec";
        setActuatorReportRateMSec((uint32_t)_config->GetConfigDoubleValue(paramStr, 23));

        paramStr = base + "MotorMaxCurrentLimitAmps";
        LinearActuatorParamsMsg->setMotorMaxCurrentLimitAmps(_config->GetConfigDoubleValue(paramStr, 65.0));

        paramStr = base + "PositionReachedErrorTimeMSec";
        LinearActuatorParamsMsg->setPositionReachedErrorTimeMSec((uint32_t)_config->GetConfigDoubleValue(paramStr, 40));

        paramStr = base + "FeedbackCtrl_KP";
        LinearActuatorParamsMsg->setFeedbackCtrl_KP((uint32_t)_config->GetConfigDoubleValue(paramStr, 1000));

        paramStr = base + "FeedbackCtrl_KI";
        LinearActuatorParamsMsg->setFeedbackCtrl_KI((uint32_t)_config->GetConfigDoubleValue(paramStr, 1000));

        paramStr = base + "FeedbackCtrl_KD";
        LinearActuatorParamsMsg->setFeedbackCtrl_KD((uint32_t)_config->GetConfigDoubleValue(paramStr, 10));

        paramStr = base + "FeedbackCtrl_CLFreq";
        LinearActuatorParamsMsg->setFeedbackCtrl_CLFreq((uint32_t)_config->GetConfigDoubleValue(paramStr, 60));

        paramStr = base + "FeedbackCtrl_ErrDeadbandInces";
        LinearActuatorParamsMsg->setFeedbackCtrl_ErrDeadbandInces(_config->GetConfigDoubleValue(paramStr, 0.05));

        paramStr = base + "Motor_MinPWM";
        LinearActuatorParamsMsg->setMotor_MinPWM((uint32_t)_config->GetConfigDoubleValue(paramStr, 20));

        paramStr = base + "Motor_MaxPWM";
        LinearActuatorParamsMsg->setMotor_MaxPWM((uint32_t)_config->GetConfigDoubleValue(paramStr, 90));

        paramStr = base + "Motor_pwmFreq";
        LinearActuatorParamsMsg->setMotor_pwmFreq((uint32_t)_config->GetConfigDoubleValue(paramStr, 2000));
    }



    /// <summary>
    /// This message is used to put the actuator in automatic mode where it
    /// controls the position or in passive mode where the shaft is free to move.
    /// The Clutch Enable and Motor Enable flags are used to control these features.
    /// In normal operation, the Clutch should be turn-on 20msec or more before the
    /// motor is enabled, and reverse when disabling the motor.
    ///
    /// Note: If the CAN actuator does not receive a
    /// command within 1 second, it will turn off both the
    /// clutch and the motor to go into a safe mode. It is
    /// suggested to refresh commands every 100ms or so.
    /// </summary>
    /// <param name="posInches"></param>
    /// <returns>false or true if error.</returns>
    void KarTechLinearActuator::generateActuatorPositionInchesMsg(struct can_frame &canMsg,
                                                                  double posInches, bool enableClutch, bool enableMotor)
    {
        canMsg.can_id = _commandID;
        canMsg.can_dlc = 8;
        int nPos = (int) (1000.0 * posInches + 0.5) + 500;
        canMsg.data[0] = 15;
        canMsg.data[1] = 10;
        canMsg.data[2] = (uint8_t) (nPos & 0xFF);
        canMsg.data[3] = (uint8_t) ((nPos >> 8) & 0x3F);
        if(enableClutch)
            canMsg.data[3] |= 0x80;
        if(enableMotor)
            canMsg.data[3] |= 0x40;

        //There is no direct feedback from the Linear Acutuator on these values
        //so set them here.  This is primarily used internal to this object
        //so there is not reason to post the message.
        LinearActuatorPositionFeedbackMsg->MotorEnable = enableMotor;
        LinearActuatorPositionFeedbackMsg->ClutchEnable = enableClutch;
        //LinearActuatorPositionFeedbackMsg->PostMessage();
    }

    void KarTechLinearActuator::generateRequestPositionAndMotorCurrentReportMsg(struct can_frame &canMsg,
                                                                                uint32_t updateRateMilliSeconds)
    {
        canMsg.can_id = _commandID;
        canMsg.can_dlc = 8;
        uint32_t timeMs = updateRateMilliSeconds > 0xFFFF ? 0xFFFF : updateRateMilliSeconds;
        canMsg.data[0] = 240;
        canMsg.data[1] = 0;
        canMsg.data[2] = 152;
        canMsg.data[3] = (uint8_t) (timeMs & 0xFF);
        canMsg.data[4] = (uint8_t) ((timeMs >> 8) & 0xFF);
        canMsg.data[5] = 255;
        canMsg.data[6] = 255;
        canMsg.data[7] = 255;
    }


    /// <summary>
    /// This message is used to set the software current limit for the motor.
    /// Normally this should be left alone. The CAN actuator hardware is self-limiting
    /// and self-protecting. However if you want to “weaken” the actuator so it
    /// cannot push too hard, you can use this message to reduce the motor
    /// current. If the motor current exceeds this setting for a little while, the motor
    /// is turned off. It will turn back on the next time it is told to move.
    /// Factory default is 65.0 amps
    /// </summary>
    /// <param name="maxMotorCurrentAmps"></param>
    /// <returns></returns>
    void KarTechLinearActuator::generateMotorOverCurrentConfigMsg(struct can_frame &canMsg,
                                                                  double maxMotorCurrentAmps)
    {
        canMsg.can_id = _commandID;
        canMsg.can_dlc = 8;
        maxMotorCurrentAmps = maxMotorCurrentAmps < 1.0 ? 1.0 : maxMotorCurrentAmps > 65.0 ? 65.0 : maxMotorCurrentAmps;
        uint32_t maxMilliAmps = (uint32_t) (1000.0 * maxMotorCurrentAmps);
        canMsg.data[0] = 126;
        canMsg.data[1] = 0x80 | 3;
        canMsg.data[2] = (uint8_t) (maxMilliAmps & 0xFF);
        canMsg.data[3] = (uint8_t) (maxMilliAmps >> 8 & 0xFF);
        canMsg.data[4] = 255;
        canMsg.data[5] = 255;
        canMsg.data[6] = 255;
        canMsg.data[7] = 255;
    }

    /// <summary>
    /// The CAN actuator can detect obstructions. When commanded to move to a
    /// particular position, and it is stopped for a certain time, an error is detected.
    /// This time is the Position Reach Error Time (PRET). The longer the time, the
    /// longer the actuator will push against an object before flagging the problem.
    /// When this error is detected, the actuator will turn off the motor for a short
    /// time, and then go full on for a short time. The intent is to try to break
    /// through any friction. Setting this value too high may reduce motor or clutch
    /// life.
    /// Default is 40msec
    /// </summary>
    /// <param name="timeLimitMilliseconds"></param>
    /// <returns></returns>
    void KarTechLinearActuator::generatePositionReachedErrorTimeMsg(struct can_frame &canMsg,
                                                                    uint32_t timeLimitMilliseconds)
    {
        canMsg.can_id = _commandID;
        canMsg.can_dlc = 8;
        timeLimitMilliseconds =
                timeLimitMilliseconds < 20 ? 20 : timeLimitMilliseconds > 500 ? 500 : timeLimitMilliseconds;
        canMsg.data[0] = 126;
        canMsg.data[1] = 0x80 | 4;
        canMsg.data[2] = (uint8_t) (timeLimitMilliseconds & 0xFF);
        canMsg.data[3] = (uint8_t) (timeLimitMilliseconds >> 8 & 0xFF);
        canMsg.data[4] = 255;
        canMsg.data[5] = 255;
        canMsg.data[6] = 255;
        canMsg.data[7] = 255;
    }

    /// <summary>
    /// This message is used to set the closed loop gain parameters KP and KI.
    /// Defaults: KP = 1000, KI = 1000
    /// </summary>
    /// <param name="Kp"></param>
    /// <param name="Ki"></param>
    /// <returns></returns>
    void KarTechLinearActuator::generateConfigure_Kp_Ki_Msg(struct can_frame &canMsg,
                                                            uint32_t KP, uint32_t KI)
    {
        canMsg.can_id = _commandID;
        canMsg.can_dlc = 8;
        KP = KP < 10 ? 10 : KP > 10000 ? 10000 : KP;
        KI = KI < 10 ? 10 : KI > 10000 ? 10000 : KI;
        canMsg.data[0] = 245;
        canMsg.data[1] = 0x80 | 1;
        canMsg.data[2] = 0;
        canMsg.data[3] = 0;
        canMsg.data[4] = (uint8_t) (KP & 0xFF);
        canMsg.data[5] = (uint8_t) (KP >> 8 & 0xFF);
        canMsg.data[6] = (uint8_t) (KI & 0xFF);
        canMsg.data[7] = (uint8_t) (KI >> 8 & 0xFF);
    }

    /// <summary>
    /// This message is used to set the closed loop gain parameters KD, the closed
    /// loop correction frequency and the Error Deadband.
    /// Defaults: KD = 10, Freq = 60, Error Deadband = 0.05 inches
    /// </summary>
    /// <param name="KP"></param>
    /// <param name="KI"></param>
    /// <returns></returns>
    void KarTechLinearActuator::generateConfigure_KD_Freq_EDB_Msg(struct can_frame &canMsg,
                                                                  uint32_t KD, uint32_t CLFreq,
                                                                  double ErrorDeadbandInches)
    {
        canMsg.can_id = _commandID;
        canMsg.can_dlc = 8;
        KD = KD < 0 ? 0 : KD > 100 ? 100 : KD;
        CLFreq = CLFreq < 20 ? 20 : CLFreq > 200 ? 200 : CLFreq;
        ErrorDeadbandInches =
                ErrorDeadbandInches < 0.005 ? 0.005 : ErrorDeadbandInches > 0.250 ? 0.250 : ErrorDeadbandInches;
        canMsg.data[0] = 245;
        canMsg.data[1] = 0x80 | 1;
        canMsg.data[2] = 0;
        canMsg.data[3] = 1;
        canMsg.data[4] = (uint8_t) (KD & 0xFF);
        canMsg.data[5] = (uint8_t) (KD >> 8 & 0xFF);
        canMsg.data[6] = (uint8_t) (CLFreq & 0xFF);
        canMsg.data[7] = (uint8_t) (1000.0 * ErrorDeadbandInches);
    }

    /// <summary>
    /// This message is used to set the Minimum Motor Drive duty cycle, the
    /// Maximum Motor Drive duty cycle, and the PWM Frequency for the motor.
    /// MIN PWM is the minimum PWM duty cycle used to drive the internal
    /// motor. The larger the value the quicker it will move, but it may
    /// overshoot and cause hunting/oscillations. Too low of a value may not
    /// let the motor break through friction, which could cause the actuator to
    /// have trouble moving smoothly.
    /// The value is from 0-100%.
    ///
    /// MAX PWM is the maximum PWM duty cycle used to drive the internal
    /// motor. The larger the value the quicker it will move, but it may
    /// overshoot and cause hunting/oscillations. Too low of a value may slow
    /// the motion too much.
    /// The value is from 0-100%.
    ///
    /// PWM FREQ is the value in Hz used for the PWM outputs
    /// Defaults: minPWM = 20%, maxPWM = 90%, pwmFreq = 2000Hz
    /// </summary>
    /// <param name="minPWM"></param>
    /// <param name="maxPWM"></param>
    /// <param name="pwmFreq"></param>
    /// <returns></returns>
    void KarTechLinearActuator::generateConfigureMotorPWM_Msg(struct can_frame &canMsg,
                                                              uint32_t minPWM, uint32_t maxPWM, uint32_t pwmFreq)
    {
        canMsg.can_id = _commandID;
        canMsg.can_dlc = 8;
        minPWM = minPWM < 0 ? 0 : minPWM > 100 ? 100 : minPWM;
        maxPWM = maxPWM < 0 ? 0 : maxPWM > 100 ? 100 : maxPWM;
        pwmFreq = pwmFreq < 1000 ? 1000 : pwmFreq > 4000 ? 4000 : pwmFreq;
        canMsg.data[0] = 245;
        canMsg.data[1] = 0x80 | 1;
        canMsg.data[2] = 0;
        canMsg.data[3] = 2;
        canMsg.data[4] = (uint8_t) (minPWM & 0xFF);
        canMsg.data[5] = (uint8_t) (maxPWM & 0xFF);
        canMsg.data[6] = (uint8_t) (pwmFreq & 0xFF);
        canMsg.data[7] = (uint8_t) (pwmFreq >> 8 & 0xFF);
    }


    /// <summary>
    /// RESET TYPE - This value determines which type of reset is to be done.
    ///    0x0001 = Reset Outputs. All outputs are turned off.
    ///    0x0002 = Reset User Defined IDs.
    ///    0x0004 = Reset Report Rates.
    ///    0x0008 = Reset Hardware Configurations.
    ///    0x0010 = Reset User Configurations (KP, KI, KD, …).
    ///    0xFFFF = Reset everything.
    ///
    /// RESET TYPE EXTENTION 1 – This parameter further defines some
    ///    resets. Setting the bit to 1, activates the specific Reset. Clearing the
    ///    bit leaves that parameter untouched.
    ///    Bit 0: Reset User Defined Report ID
    ///    Bit 1: Reset User Defined Command ID #1
    ///    Bit 2: Reset User Defined Command ID #2
    ///    Bit 3: Reset User Defined Command ID #3
    ///    Bit 4: Reset User Defined Command ID #4
    ///    Bit 13: Reset RPSEL.
    ///    Bit 14: Reset DISDEF
    /// </summary>
    /// <param name="resetTypeFlags"></param>
    /// <param name="resetExtFlags"></param>
    /// <returns></returns>
    void KarTechLinearActuator::generateResetActuatorMsg(struct can_frame &canMsg,
                                                         uint32_t resetTypeFlags, uint32_t resetExtFlags)
    {
        canMsg.can_id = _commandID;
        canMsg.can_dlc = 8;
        canMsg.data[0] = 249;
        canMsg.data[1] = 0;
        canMsg.data[2] = (uint8_t) (resetTypeFlags & 0xFF);
        canMsg.data[3] = (uint8_t) (resetTypeFlags >> 8 & 0xFF);
        canMsg.data[4] = (uint8_t) (resetExtFlags & 0xFF);
        canMsg.data[5] = (uint8_t) (resetExtFlags >> 8 & 0xFF);
        canMsg.data[6] = 0;
        canMsg.data[7] = 0;
    }


    /// <summary>
    /// This request the KarTech Acuator reset and calibrate the internal position sensor.
    /// The actualtor shaft must be free to move over its full range... so only use this
    /// command when the actuator is disconnected from the system.
    /// </summary>
    /// <returns></returns>
    void KarTechLinearActuator::generateAutoZeroCalibrationMsg(struct can_frame &canMsg)
    {
        canMsg.can_id = _commandID;
        canMsg.can_dlc = 8;
        canMsg.data[0] = 126;
        canMsg.data[1] = 2;
        canMsg.data[2] = 18;
        canMsg.data[3] = 52;
        canMsg.data[4] = 86;
        canMsg.data[5] = 171;
        canMsg.data[6] = 205;
        canMsg.data[7] = 239;
    }


    //This command configures the User Defined Report CAN ID.  It is also used to
    //switch between the user defined on and the default ID.
    //When setting up a new KarTech Actuator, the default command ID is used.  In this case
    //only one KarTech Actuator should be connected to the CAN bus, otherwise all KarTech Actuators
    //will be set the same.
    void KarTechLinearActuator::generateSetReportIDMsg(struct can_frame &canMsg, bool useDefaultCmdId,
                                                         uint32_t reportID, bool useUserDefinedID)
    {
        canMsg.can_id = useDefaultCmdId ? DefaultCmdID : _commandID;
        canMsg.can_dlc = 8;
        canMsg.data[0] = 247;
        canMsg.data[1] = 0x80;
        canMsg.data[2] = (uint8_t) (reportID & 0xFF);
        canMsg.data[3] = (uint8_t) (reportID >> 8  & 0xFF);
        canMsg.data[4] = (uint8_t) (reportID >> 16 & 0xFF);
        canMsg.data[5] = (uint8_t) (reportID >> 24 & 0xFF);
        canMsg.data[6] = useUserDefinedID? 1 : 0;
        canMsg.data[7] = 0;
    }


    //This command configures the User Defined Command CAN ID.
    //When setting up a new KarTech Actuator, the default command ID is used.  In this case
    //only one KarTech Actuator should be connected to the CAN bus, otherwise all KarTech Actuators
    //will be set the same.
    void KarTechLinearActuator::generateSetCommandIDMsg(struct can_frame &canMsg, uint32_t cmdID)
    {
        canMsg.can_id = DefaultCmdID;
        canMsg.can_dlc = 8;
        canMsg.data[0] = 247;
        canMsg.data[1] = 0x80 | 0x01;
        canMsg.data[2] = (uint8_t) (cmdID & 0xFF);
        canMsg.data[3] = (uint8_t) (cmdID >> 8  & 0xFF);
        canMsg.data[4] = (uint8_t) (cmdID >> 16 & 0xFF);
        canMsg.data[5] = (uint8_t) (cmdID >> 24 & 0xFF);
        canMsg.data[6] = 0;
        canMsg.data[7] = 0;
    }


    /// <summary>
    /// This request poles up to 6 messages/replies at once. Send this and the
    /// actuator will reply with up to 6 different messages.
    /// </summary>
    /// <param name="msgIdx1"></param>
    /// <param name="msgIdx2"></param>
    /// <param name="msgIdx3"></param>
    /// <param name="msgIdx4"></param>
    /// <param name="msgIdx5"></param>
    /// <param name="msgIdx6"></param>
    /// <returns></returns>
    void KarTechLinearActuator::generateRequestReportMsg(struct can_frame &canMsg, uint8_t msgIdx1,
                                                         uint8_t msgIdx2, uint8_t msgIdx3, uint8_t msgIdx4,
                                                         uint8_t msgIdx5, uint8_t msgIdx6)
    {
        canMsg.can_id = _commandID;
        canMsg.can_dlc = 8;
        canMsg.data[0] = 41;
        canMsg.data[1] = 0;
        canMsg.data[2] = msgIdx1;
        canMsg.data[3] = msgIdx2;
        canMsg.data[4] = msgIdx3;
        canMsg.data[5] = msgIdx4;
        canMsg.data[6] = msgIdx5;
        canMsg.data[7] = msgIdx6;
    }


    void KarTechLinearActuator::logKTLAStatusData()
    {
        if( CanRxDataRecorderPtr != nullptr && LogActuatorPostionFeedbackData)
        {
            bool logMsgChanged = _loggingControlMsg->FetchMessage();
            if(_loggingControlMsg->EnableLogging)
            {
                CanRxDataRecorderPtr->writeDataRecord(KTLAStatusDataRecord);
            }
            else if(logMsgChanged && !_loggingControlMsg->EnableLogging)
            {
                CanRxDataRecorderPtr->closeLogFile();
            }
        }
    }

    void KarTechLinearActuator::processRxCanMessage(struct can_frame &canMsg)
    {
        uint32_t val;
        double dval;
        double posInches = 0;
        double posPercent = 0;
        double currentAmps = 0;
        int UnknownReportMsgIdx = 0;

        switch(canMsg.data[0])
        {
            case 126:
                if(((int) canMsg.data[1] & 0x3F) == 3)
                {
                    //Motor Max Current Confermation Message
                    val = (uint32_t) canMsg.data[3] << 8;
                    val |= (uint32_t) canMsg.data[2];
                    double maxCurrent = 0.001 * (double) val;
                    LinearActuatorParamsFeedbackMsg->setMotorMaxCurrentLimitAmps(maxCurrent);
                    LinearActuatorParamsFeedbackMsg->PostMessage();
                    logKTLAStatusData();
                }
                if(((int) canMsg.data[1] & 0x3F) == 4)
                {
                    val = (uint32_t) canMsg.data[3] << 8;
                    val |= (uint32_t) canMsg.data[2];
                    LinearActuatorParamsFeedbackMsg->setPositionReachedErrorTimeMSec(val);
                    LinearActuatorParamsFeedbackMsg->PostMessage();
                    logKTLAStatusData();
                }
                break;

            case 127:
                val = (uint32_t) canMsg.data[3] << 8;
                val |= (uint32_t) canMsg.data[2];
                dval = 0.001 * (double)(val - 500);
                KTLAStatusDataRecord.ActuatorPostionInches = dval;
                posPercent = postionInchesToPercent(dval);
                LinearActuatorPositionFeedbackMsg->setPositionPercent(posPercent);
                LinearActuatorPositionFeedbackMsg->PostMessage();
                logKTLAStatusData();
                break;

            case 129:
                //Motor Current and Temperature
                val = (uint32_t) canMsg.data[3] << 8;
                val |= (uint32_t) canMsg.data[2];
                dval = 0.001 * (double)(val);
                LinearActuatorPositionFeedbackMsg->setMotorCurrentAmps(dval);

                val = (uint32_t) canMsg.data[5] << 8;
                val |= (uint32_t) canMsg.data[4];
                dval = 0.1 * (double)(val);
                if(canMsg.data[6] == 1)
                    dval = -dval;  //Neg. Temp
                LinearActuatorPositionFeedbackMsg->TempDegC = dval;
                LinearActuatorPositionFeedbackMsg->PostMessage();
                logKTLAStatusData();
                break;

            case 152:
                val = (uint32_t) canMsg.data[3] << 8;
                val |= (uint32_t) canMsg.data[2];
                dval = 0.001 * (double)(val - 500);
                KTLAStatusDataRecord.ActuatorPostionInches = dval;
                posPercent = postionInchesToPercent(dval);
                LinearActuatorPositionFeedbackMsg->setPositionPercent(posPercent);

                val = (uint32_t) canMsg.data[6] << 8;
                val |= (uint32_t) canMsg.data[5];
                dval = 0.001 * (double)(val);
                LinearActuatorPositionFeedbackMsg->setMotorCurrentAmps(dval);
                LinearActuatorPositionFeedbackMsg->ErrorFlags = canMsg.data[4];
                LinearActuatorPositionFeedbackMsg->Status = canMsg.data[7];
                LinearActuatorPositionFeedbackMsg->PostMessage();
                logKTLAStatusData();
                break;

            case 245:
                if(((int) canMsg.data[1] & 0x3F) == 1
                   && canMsg.data[2] == 0 && canMsg.data[3] == 0)
                {
                    val = (uint32_t) canMsg.data[5] << 8;
                    val |= (uint32_t) canMsg.data[4];
                    LinearActuatorParamsFeedbackMsg->setFeedbackCtrl_KP(val);
                    val = (uint32_t) canMsg.data[7] << 8;
                    val |= (uint32_t) canMsg.data[6];
                    LinearActuatorParamsFeedbackMsg->setFeedbackCtrl_KI(val);
                    LinearActuatorParamsFeedbackMsg->PostMessage();
                }
                if(((int) canMsg.data[1] & 0x3F) == 1
                   && canMsg.data[2] == 0 && canMsg.data[3] == 1)
                {
                    val = (uint32_t) canMsg.data[5] << 8;
                    val |= (uint32_t) canMsg.data[4];
                    LinearActuatorParamsFeedbackMsg->setFeedbackCtrl_KD(val);
                    LinearActuatorParamsFeedbackMsg->setFeedbackCtrl_CLFreq((uint32_t)canMsg.data[6]);
                    LinearActuatorParamsFeedbackMsg->setFeedbackCtrl_ErrDeadbandInces((double)canMsg.data[7] * 0.001);
                    LinearActuatorParamsFeedbackMsg->PostMessage();
                }
                if(((int) canMsg.data[1] & 0x3F) == 1
                   && canMsg.data[2] == 0 && canMsg.data[3] == 2)
                {
                    //Motor Max Current Confermation Message
                    LinearActuatorParamsFeedbackMsg->setMotor_MinPWM((uint32_t)canMsg.data[4]);
                    LinearActuatorParamsFeedbackMsg->setMotor_MaxPWM((uint32_t)canMsg.data[5]);
                    val = (uint32_t) canMsg.data[7] << 8;
                    val |= (uint32_t) canMsg.data[6];
                    LinearActuatorParamsFeedbackMsg->setMotor_pwmFreq(val);
                    LinearActuatorParamsFeedbackMsg->PostMessage();
                }
                if(((int) canMsg.data[1] & 0x3F) == 1
                   && canMsg.data[2] == 0 && canMsg.data[3] == 3)
                {
                    //Motor Max Current Confermation Message
                    //ActuatorTempOffsetDegC = (uint32_t)canMsg.data[4];
                }
                break;

            case 247:
                if(((int) canMsg.data[1] & 0x3F) == 0)
                {
                    //Actuator User Defined Report ID
                    val = (uint32_t) canMsg.data[2];
                    val |= (uint32_t) canMsg.data[3] << 8;
                    val |= (uint32_t) canMsg.data[4] << 16;
                    val |= (uint32_t) canMsg.data[5] << 24;
                    //_reportID = val;
                }
                if(((int) canMsg.data[1] & 0x3F) == 1)
                {
                    //Actuator User Defined Report ID
                    val = (uint32_t) canMsg.data[2];
                    val |= (uint32_t) canMsg.data[3] << 8;
                    val |= (uint32_t) canMsg.data[4] << 16;
                    val |= (uint32_t) canMsg.data[5] << 24;
                    //_commandID = val;
                }
                break;

            default:
                UnknownReportMsgIdx = canMsg.data[0];
                break;
        }
    }

}

    
