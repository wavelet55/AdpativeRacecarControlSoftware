using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CarCANBusMonitor.Widgets;
using FalconVisionMonitorViewer;
using VisionBridge.Messages;

namespace CarCANBusMonitor
{

    public enum LinearActuatorFunction_e
    {
        Default,
        Brake,
        Accelerator,
    }


    public class KarTeckLinearActuator
    {
        public VisionCmdProcess VisionCmdProc;

        public const UInt32 DefaultCmdID = 0x00FF0000;
        public const UInt32 DefaultRptID = 0x00FF0001;

        public const UInt32 BrakeCmdID = 0x00550100;
        public const UInt32 BrakeRptID = 0x00550101;

        public const UInt32 AcceleratorCmdID = 0x00550200;
        public const UInt32 AcceleratorRptID = 0x00550201;

        private UInt32 _commandID = DefaultCmdID;
        public UInt32 CommandID
        {
            get { return _commandID; }
        }

        private UInt32 _reportID = DefaultRptID;
        public UInt32 ReportID
        {
            get { return _reportID; }
        }

        public bool IsRunning
        {
            get { return true; }
        }

        private LinearActuatorFunction_e _linearActuatorFunction;
        public LinearActuatorFunction_e LinearActuatorFunction
        {
            get { return _linearActuatorFunction; }
        }


        private double _minPositionInches = 0.0;
        public double MinPositionInches
        {
            get { return _minPositionInches; }
            set { _minPositionInches = value < 0 ? 0 : value > 2.9 ? 2.9 : value; }
        }

        private double _maxPositionInches = 3.0;
        public double MaxPositionInches
        {
            get { return _maxPositionInches; }
            set { _maxPositionInches = value < 0.1 ? 0.1 : value > 3.0 ? 3.0 : value; }
        }

        public double PositionRangeInches
        {
            get
            {
                double range = MaxPositionInches - MinPositionInches;
                range = range < 0.10 ? 0.10 : range;
                return range;
            }
        }

        //Set position in percent... this is what Videre Accepts.
        //The actual position in inches can be computed from this based upon 
        //the Min and Max values and whether the actuator is setup for push or pull control.
        private double _desiredPositionPercent = 0.0;
        public double DesiredPositionPercent
        {
            get
            {
                return _desiredPositionPercent;
            }
            set
            {
                _desiredPositionPercent = value < 0 ? 0 : value > 100.0 ? 100.0 : value;
            }
        }

        public double CurrentPositionPercent = 0.0;

        public bool EnableClutchMc = false;
        public bool EnableMotorMc = false;

        public bool ClutchEnabledFb = false;
        public bool MotorEnabledFb = false;

        public bool ManualControlCmd = false;
        public bool ManualControlFb = false;

        public bool SetupModeCmd = false;
        public bool SetupModeFb = false;

        //Current Motor Current... this is dynamic... so it will typically be zero.
        public double MotorCurrentAmps = 0;

        public double MaxMotorCurrentAmps = 0;

        public double SetMaxMotorCurrentAmps
        {
            set { MaxMotorCurrentAmps = value > MaxMotorCurrentAmps ? value : MaxMotorCurrentAmps; }
        }

        public void ClearMaxMotorCurrent()
        {
            MaxMotorCurrentAmps = 0;
        }

        //Note: the max measured current is less thatn 6.0 amps... it appears that
        //this Current Limit is off by a factor of 10.  The value should not be changed
        //so leave it alone.
        public double MotorMaxCurrentLimitAmps = 65.0;

        public UInt32 PositionReachedErrorTimeMSec = 40;

        public UInt32 FeedbackCtrl_KP = 1000;
        public UInt32 FeedbackCtrl_KI = 1000;
        public UInt32 FeedbackCtrl_KD = 10;
        public UInt32 FeedbackCtrl_CLFreq = 60;
        public double FeedbackCtrl_ErrDeadbandInches = 0.05;

        //Motor PWM Settings
        public UInt32 Motor_MinPWM = 20;
        public UInt32 Motor_MaxPWM = 90;
        public UInt32 Motor_pwmFreq = 2000;

        public UInt32 ActuatorTempOffsetDegC = 22;

        public double ActuatorTempDegC = 22;

        public byte AcuatorError = 0;
        public byte AcuatorStatus = 0;
        public byte AcuatorTempStatus = 0;

        public bool AcuatorPostionInfoChanged = false;
        public bool AcuatorSetupInfoChanged = false;

        public byte UnknownReportMsgIdx = 0;

        public KarTeckLinearActuator(LinearActuatorFunction_e fnType, VisionCmdProcess visionCmdProc )
        {
            VisionCmdProc = visionCmdProc;
            setActuatorType(fnType);

            //ToDo:  Add a config file with the values we would like to 
            //set.
            SetFactoryDefaultValues();
        }

        public static UInt32 getCommandIDForActuatorFunction(LinearActuatorFunction_e actType)
        {
            UInt32 id = DefaultCmdID;
            switch (actType)
            {
                case LinearActuatorFunction_e.Accelerator:
                    id = AcceleratorCmdID;
                    break;

                case LinearActuatorFunction_e.Brake:
                    id = BrakeCmdID;
                    break;

                default:
                    id = DefaultCmdID;
                    break;
            }
            return id;
        }

        public static UInt32 getReportIDForActuatorFunction(LinearActuatorFunction_e actType)
        {
            UInt32 id = DefaultRptID;
            switch (actType)
            {
                case LinearActuatorFunction_e.Accelerator:
                    id = AcceleratorRptID;
                    break;

                case LinearActuatorFunction_e.Brake:
                    id = BrakeRptID;
                    break;

                default:
                    id = DefaultRptID;
                    break;
            }
            return id;
        }

        public void setActuatorType(LinearActuatorFunction_e actType)
        {
            switch (actType)
            {
                case LinearActuatorFunction_e.Accelerator:
                    _linearActuatorFunction = LinearActuatorFunction_e.Accelerator;
                    _commandID = AcceleratorCmdID;
                    _reportID = AcceleratorRptID;
                    break;

                case LinearActuatorFunction_e.Brake:
                    _linearActuatorFunction = LinearActuatorFunction_e.Brake;
                    _commandID = BrakeCmdID;
                    _reportID = BrakeRptID;
                    break;

                default:
                    _linearActuatorFunction = LinearActuatorFunction_e.Default;
                    _commandID = DefaultCmdID;
                    _reportID = DefaultRptID;
                    break;
            }
        }

        //These are the published Factory default values.
        //There does not seem to be a way to read the stored values other
        //than setting them and monitoring the result message.
        public void SetFactoryDefaultValues()
        {
            MinPositionInches = 0;
            MaxPositionInches = 3.0;

            MotorMaxCurrentLimitAmps = 65.0;
            PositionReachedErrorTimeMSec = 40;

            FeedbackCtrl_KP = 1000;
            FeedbackCtrl_KI = 1000;
            FeedbackCtrl_KD = 10;
            FeedbackCtrl_CLFreq = 60;
            FeedbackCtrl_ErrDeadbandInches = 0.05;

            Motor_MinPWM = 20;
            Motor_MaxPWM = 90;
            Motor_pwmFreq = 2000;

            ActuatorTempOffsetDegC = 22;
        }


        //Reset Actuator so that it is in the disabled
        //state and position is at minimum...
        public void ResetActuator()
        {
            ManualControlCmd = false;
            SetupModeCmd = false;
            EnableMotorMc = false;
            EnableClutchMc = false;
            DesiredPositionPercent = 0.0;
            sendPositionPercent();
        }

        /// <summary>
        /// This message is used to put the actuator in automatic mode where it
        /// controls the position or in passive mode where the shaft is free to move.
        /// The Clutch Enable and Motor Enable flags are used to control these features.
        /// In normal operation, the Clutch should be turn-on 20msec or more before the 
        /// motor is enabled, and reverse when disabling the motor.
        /// </summary>
        /// <param name="posInches"></param>
        /// <returns>false or true if error.</returns>
        public bool sendPositionPercent()
        {
            bool error = true;
            LinearActuatorPositionCtrlPBMsg laCtrlMsg = new LinearActuatorPositionCtrlPBMsg();
            if (SetupModeCmd)
            {
                laCtrlMsg.ActuatorSetupMode = true;
                laCtrlMsg.ManualExtControl = false;
                laCtrlMsg.MotorEnable = false;
                laCtrlMsg.ClutchEnable = false;
                laCtrlMsg.PositionPercent = 0;
            }
            else
            {
                laCtrlMsg.ActuatorSetupMode = false;
                laCtrlMsg.ManualExtControl = ManualControlCmd;
                laCtrlMsg.MotorEnable = EnableMotorMc;
                laCtrlMsg.ClutchEnable = EnableClutchMc;
                laCtrlMsg.PositionPercent = DesiredPositionPercent;
            }
            VisionCmdProc.SendLinearActuatorPositionMsgOnCmdPort(laCtrlMsg, _linearActuatorFunction);
            return error;
        }

        public bool sendKarTeckActuatorConfigParams()
        {
            KarTechLinearActuatorParamsPBMsg cfgMsg = new KarTechLinearActuatorParamsPBMsg();
            cfgMsg.MinPositionInches = MinPositionInches;
            cfgMsg.MaxPositionInches = MaxPositionInches;
            cfgMsg.MotorMaxCurrentLimitAmps = 65.0;  //Force to defalt
            cfgMsg.FeedbackCtrl_ErrDeadbandInches = FeedbackCtrl_ErrDeadbandInches;
            cfgMsg.FeedbackCtrl_KP = FeedbackCtrl_KP;
            cfgMsg.FeedbackCtrl_KI = FeedbackCtrl_KI;
            cfgMsg.FeedbackCtrl_KD = FeedbackCtrl_KD;
            cfgMsg.FeedbackCtrl_CLFreq = FeedbackCtrl_CLFreq;
            cfgMsg.Motor_MinPWM = Motor_MinPWM;
            cfgMsg.Motor_MaxPWM = Motor_MaxPWM;
            cfgMsg.Motor_pwmFreq = Motor_pwmFreq;
            cfgMsg.PositionReachedErrorTimeMSec = PositionReachedErrorTimeMSec;
            VisionCmdProc.SendKarTeckActuatorConfigMsgOnCmdPort(cfgMsg, _linearActuatorFunction);
            return false;
        }

        public void sendAutoZeroCalCommand()
        {
            KarTechLinearActuatorSetupPBMsg msg = new KarTechLinearActuatorSetupPBMsg();
            msg.AutoZeroCal = true;
            VisionCmdProc.SendKarTeckActuatorSetupMsgOnCmdPort(msg, _linearActuatorFunction);
        }

        public void sendOutputResetCommand()
        {
            KarTechLinearActuatorSetupPBMsg msg = new KarTechLinearActuatorSetupPBMsg();
            msg.ResetOutputs = true;
            VisionCmdProc.SendKarTeckActuatorSetupMsgOnCmdPort(msg, _linearActuatorFunction);
        }

        public void sendHardwareConfigResetCommand()
        {
            KarTechLinearActuatorSetupPBMsg msg = new KarTechLinearActuatorSetupPBMsg();
            msg.ResetHardwareCfgs = true;
            VisionCmdProc.SendKarTeckActuatorSetupMsgOnCmdPort(msg, _linearActuatorFunction);
        }

        public void sendUserConfigResetCommand()
        {
            KarTechLinearActuatorSetupPBMsg msg = new KarTechLinearActuatorSetupPBMsg();
            msg.ResetUserCfgs = true;
            VisionCmdProc.SendKarTeckActuatorSetupMsgOnCmdPort(msg, _linearActuatorFunction);
        }

        public void sendResetAllCommand()
        {
            KarTechLinearActuatorSetupPBMsg msg = new KarTechLinearActuatorSetupPBMsg();
            msg.ResetAll = true;
            VisionCmdProc.SendKarTeckActuatorSetupMsgOnCmdPort(msg, _linearActuatorFunction);
        }

        public void sendSetComandResponseIDsCommand()
        {
            KarTechLinearActuatorSetupPBMsg msg = new KarTechLinearActuatorSetupPBMsg();
            msg.SetCanCommandResponseIDs = true;
            VisionCmdProc.SendKarTeckActuatorSetupMsgOnCmdPort(msg, _linearActuatorFunction);
        }

        public bool getKarTeckActuatorConfigParams()
        {
            bool retrievedCfgParams = false;
            string resultsMsg;
            KarTechLinearActuatorParamsPBMsg cfgMsg = VisionCmdProc.GetKarTeckActuatorConfigSettings(_linearActuatorFunction, out resultsMsg);
            if (cfgMsg != null)
            {
                MinPositionInches = cfgMsg.MinPositionInches;
                MaxPositionInches = cfgMsg.MaxPositionInches;
                MotorMaxCurrentLimitAmps = cfgMsg.MotorMaxCurrentLimitAmps;  //Force to defalt
                FeedbackCtrl_ErrDeadbandInches = cfgMsg.FeedbackCtrl_ErrDeadbandInches;
                FeedbackCtrl_KP = cfgMsg.FeedbackCtrl_KP;
                FeedbackCtrl_KI = cfgMsg.FeedbackCtrl_KI;
                FeedbackCtrl_KD = cfgMsg.FeedbackCtrl_KD;
                FeedbackCtrl_CLFreq = cfgMsg.FeedbackCtrl_CLFreq;
                Motor_MinPWM = cfgMsg.Motor_MinPWM = Motor_MinPWM;
                Motor_MaxPWM = cfgMsg.Motor_MaxPWM = Motor_MaxPWM;
                Motor_pwmFreq = cfgMsg.Motor_pwmFreq;
                PositionReachedErrorTimeMSec = cfgMsg.PositionReachedErrorTimeMSec;
                retrievedCfgParams = true;
            }
            return retrievedCfgParams;
        }

        public void processPositionStatusMsg(LinearActuatorPositionCtrlPBMsg statusMsg)
        {
            CurrentPositionPercent = statusMsg.PositionPercent;
            MotorCurrentAmps = statusMsg.MotorCurrentAmps;
            //Capture Max Current
            if (MotorCurrentAmps > MaxMotorCurrentAmps)
                MaxMotorCurrentAmps = MotorCurrentAmps;

            ActuatorTempDegC = statusMsg.TempDegC;
            AcuatorError = (byte)statusMsg.ErrorFlags;
            ClutchEnabledFb = statusMsg.ClutchEnable;
            MotorEnabledFb = statusMsg.MotorEnable;
            ManualControlFb = statusMsg.ManualExtControl;
            SetupModeFb = statusMsg.ActuatorSetupMode;
        }

    }
}
