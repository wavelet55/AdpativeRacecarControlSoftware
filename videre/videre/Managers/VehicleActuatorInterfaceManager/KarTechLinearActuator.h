/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: June 1, 2018
 *
 * KarTech Linear Actuator Interface
 *******************************************************************/

#ifndef VIDERE_DEV_KARTECHLINEARACTUATOR_H
#define VIDERE_DEV_KARTECHLINEARACTUATOR_H

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <linux/can.h>
#include <RabitManager.h>
#include "global_defines.h"
#include "../../Utils/logger.h"
#include "../../Utils/timing.h"
#include "../../Utils/config_data.h"
#include "KarTechLinearActuatorParamsMessage.h"
#include "KarTechLinearActuatorSetupMessage.h"
#include "LinearActuatorPositionCtrlMessage.h"
#include "DataRecorder.h"
#include "KarTechLADataRecords.h"
#include "ImageLoggingControlMessage.h"

namespace videre
{


    class KarTechLinearActuator
    {

    public:
        const uint32_t DefaultCmdID = 0x00FF0000;
        const uint32_t DefaultRptID = 0x00FF0001;

        const uint32_t BrakeCmdID = 0x00550100;
        const uint32_t BrakeRptID = 0x00550101;

        const uint32_t AcceleratorCmdID = 0x00550200;
        const uint32_t AcceleratorRptID = 0x00550201;

        const double MinActuatorAbsolutePositionInches = 0.0;
        const double MaxActuatorAbsolutePositionInches = 3.0;

        std::shared_ptr<LinearActuatorPositionCtrlMessage> PositionControlMsg;
        std::shared_ptr<LinearActuatorPositionCtrlMessage> LinearActuatorPositionFeedbackMsg;

        std::shared_ptr<KarTechLinearActuatorParamsMessage> LinearActuatorParamsControlMsg;
        std::shared_ptr<KarTechLinearActuatorParamsMessage> LinearActuatorParamsFeedbackMsg;

        std::shared_ptr<KarTechLinearActuatorSetupMessage> LinearActuatorSetupMsg;

        std::shared_ptr<ImageLoggingControlMessage> _loggingControlMsg;

        DataRecorder *CanRxDataRecorderPtr = nullptr;
        KarTechLAStatusDataRecord KTLAStatusDataRecord;

        bool LogActuatorPostionFeedbackData = true;


    private:

        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        //A reference to the CommManger... primarily used during setup of this
        //class object.
        Rabit::RabitManager *_mgrPtr;
        std::shared_ptr<ConfigData> _config;

        LinearActuatorFunction_e _functionType;
        LinearActuatorPullPushType_e _pushPullType;
        uint32_t _commandID = DefaultCmdID;
        uint32_t _reportID = DefaultRptID;

        uint32_t _actuatorReportRateMSec = 49;

        double _minActuatorPositionInches = 0.0;
        double _maxActuatorPositionInches = 3.0;

    public:

        KarTechLinearActuator(Rabit::RabitManager *_mgrPtr,
                              LinearActuatorFunction_e functionType,
                              std::shared_ptr<ConfigData> config);

        LinearActuatorFunction_e getFuctionType()
        { return _functionType; }

        LinearActuatorPullPushType_e getActuatorPullPushType()  {return _pushPullType;}

        uint32_t getCanCmdID()
        { return _commandID; }

        uint32_t getCanRptID()
        { return _reportID; }

        uint32_t getActuatorReportRateMSec() {return _actuatorReportRateMSec;}
        void setActuatorReportRateMSec(uint32_t value)
        {
            _actuatorReportRateMSec = value < 5 ? 5 : value > 1000 ? 1000 : value;
        }

        double getMinActuatorPositionInches()  { return _minActuatorPositionInches; }
        void setMinActuatorPositionInches(double val)
        {
            _minActuatorPositionInches = val < MinActuatorAbsolutePositionInches ? MinActuatorAbsolutePositionInches
                    : val > MaxActuatorAbsolutePositionInches ? MaxActuatorAbsolutePositionInches : val;
        }

        double getMaxActuatorPositionInches()  { return _maxActuatorPositionInches; }
        void setMaxActuatorPositionInches(double val)
        {
            _maxActuatorPositionInches = val < MinActuatorAbsolutePositionInches ? MinActuatorAbsolutePositionInches
                                                                                 : val > MaxActuatorAbsolutePositionInches ? MaxActuatorAbsolutePositionInches : val;
        }

        double getActuatorRangeInches()
        {
            double range = _maxActuatorPositionInches - _minActuatorPositionInches;
            range = range < 0.1 ? 0.1 : range;
            return range;
        }

        double postionPercentToInches(double valPercent);

        //This will may return values outside of the range of [0, 100.0]
        //since the acuatal position can be outside the min max values.
        double postionInchesToPercent(double posInches);

        //Read the KarTech Config Parameters from the config file and store them in the
        //LinearActuatorParamsMsg.
        void readParametersConfig(std::shared_ptr<KarTechLinearActuatorParamsMessage> LinearActuatorParamsMsg);

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
        void generateActuatorPositionInchesMsg(struct can_frame &canMsg,
                                      double posInches, bool enableClutch, bool enableMotor);

        void generateRequestPositionAndMotorCurrentReportMsg(struct can_frame &canMsg,
                                                    uint32_t updateRateMilliSeconds);

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
        void generateMotorOverCurrentConfigMsg(struct can_frame &canMsg,
                                               double maxMotorCurrentAmps);

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
        void generatePositionReachedErrorTimeMsg(struct can_frame &canMsg,
                                         uint32_t timeLimitMilliseconds);


        /// <summary>
        /// This message is used to set the closed loop gain parameters KP and KI.
        /// Defaults: KP = 1000, KI = 1000
        /// </summary>
        /// <param name="Kp"></param>
        /// <param name="Ki"></param>
        /// <returns></returns>
        void generateConfigure_Kp_Ki_Msg(struct can_frame &canMsg,
                             uint32_t KP, uint32_t KI);


        /// <summary>
        /// This message is used to set the closed loop gain parameters KD, the closed
        /// loop correction frequency and the Error Deadband.
        /// Defaults: KD = 10, Freq = 60, Error Deadband = 0.05 inches
        /// </summary>
        /// <param name="KP"></param>
        /// <param name="KI"></param>
        /// <returns></returns>
        void generateConfigure_KD_Freq_EDB_Msg(struct can_frame &canMsg,
                                               uint32_t KD, uint32_t CLFreq, double ErrorDeadbandInches);

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
        void generateConfigureMotorPWM_Msg(struct can_frame &canMsg,
                                    uint32_t minPWM, uint32_t maxPWM, uint32_t pwmFreq);


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
        void generateResetActuatorMsg(struct can_frame &canMsg,
                uint32_t resetTypeFlags, uint32_t resetExtFlags = 0);

        /// <summary>
        /// This request the KarTech Acuator reset and calibrate the internal position sensor.
        /// The actualtor shaft must be free to move over its full range... so only use this
        /// command when the actuator is disconnected from the system.
        /// </summary>
        /// <returns></returns>
        void generateAutoZeroCalibrationMsg(struct can_frame &canMsg);

        //This command configures the User Defined Report CAN ID.  It is also used to
        //switch between the user defined on and the default ID.
        //When setting up a new KarTech Actuator, the default command ID is used.  In this case
        //only one KarTech Actuator should be connected to the CAN bus, otherwise all KarTech Actuators
        //will be set the same.
        void generateSetReportIDMsg(struct can_frame &canMsg, bool useDefaultCmdId,
                                     uint32_t reportID, bool useUserDefinedID);


        //This command configures the User Defined Command CAN ID.
        //When setting up a new KarTech Actuator, the default command ID is used.  In this case
        //only one KarTech Actuator should be connected to the CAN bus, otherwise all KarTech Actuators
        //will be set the same.
        void generateSetCommandIDMsg(struct can_frame &canMsg, uint32_t cmdID);



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
        void generateRequestReportMsg(struct can_frame &canMsg, uint8_t msgIdx1,
                                          uint8_t msgIdx2 = 0xff, uint8_t msgIdx3 = 0xff,
                                          uint8_t msgIdx4 = 0xff, uint8_t msgIdx5 = 0xff,
                                          uint8_t msgIdx6 = 0xff);


        void processRxCanMessage(struct can_frame &canMsg);

        void logKTLAStatusData();

    };

}
#endif //VIDERE_DEV_KARTECHLINEARACTUATOR_H
