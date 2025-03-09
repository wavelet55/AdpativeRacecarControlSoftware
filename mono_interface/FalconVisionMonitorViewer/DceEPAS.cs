using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FalconVisionMonitorViewer;
using VisionBridge.Messages;

namespace CarCANBusMonitor
{
    public class DceEPAS
    {
        public VisionCmdProcess VisionCmdProc;

        public const UInt32 SteeringTorqueRptID = 0x0290;
        public const UInt32 SteeringAngleRptID = 0x0292;
        public const UInt32 SteeringControlID = 0x0296;

        public double MotorCurrentAmps = 0;
        public double PWMDutyCyclePercent = 0;
        public double MotorTorque = 0;
        public double SupplyVoltage = 0;
        public double TempDegC = 0;
        public UInt32 SwitchPosition = 0;      //0 - 15
        public int TorqueA = 0;
        public int TorqueB = 0;

        public double SteeringAngleDeg = 0;     //0 degrees is neutral position

        public UInt32 TorqueMapActual = 0;

        private UInt32 _selectedTorqueMap = 0;
        public UInt32 SelectedTorqueMap      //0 - 5
        {
            get { return _selectedTorqueMap; }
            set { _selectedTorqueMap = value < 0 ? 0 : value > 5 ? 5 : value; }
        }

        private double _ctrlSteeringTorquePercent = 0;   //-100 to + 100;
        public double CtrlSteeringTorquePercent
        {
            get { return _ctrlSteeringTorquePercent; }
            set { _ctrlSteeringTorquePercent = value < -100.0 ? -100.0 : value > 100.0 ? 100.0 : value; }
        }

        public UInt32 ErrorCode = 0;
        public UInt32 StatusFlags = 0;
        public UInt32 LimitFlags = 0;

        public bool IsProgramPaused
        {
            get { return (StatusFlags & 0x01) != 0; }
        }
        public bool IsMotorMovingRight
        {
            get { return (StatusFlags & 0x02) != 0; }
        }
        public bool IsMotorMovingLeft
        {
            get { return (StatusFlags & 0x04) != 0; }
        }
        public bool IsHostModeActive
        {
            get { return (StatusFlags & 0x08) != 0; }
        }
        public bool IsFaultLight
        {
            get { return (StatusFlags & 0x10) != 0; }
        }

        public bool IsSteeringAtLeftHandStop
        {
            get { return (LimitFlags & 0x01) != 0; }
        }
        public bool IsSteeringAtRightHandStop
        {
            get { return (LimitFlags & 0x02) != 0; }
        }
        public bool IsOverTemp
        {
            get { return (LimitFlags & 0x03) != 0; }
        }

        public bool ManualControlCmd = false;
        public bool ManualControlFb = false;

        public bool EnableSteeringControl = false;

        public DceEPAS(VisionCmdProcess visionCmdProc) 
        {
            VisionCmdProc = visionCmdProc;
        }

 
        public bool sendSteeringTorqueControl()
        {
            bool error = false;
            SteeringTorqueCtrlPBMsg stcMsg = new SteeringTorqueCtrlPBMsg();
            stcMsg.ManualExtControl = ManualControlCmd;
            if(ManualControlCmd && EnableSteeringControl)
            {
                stcMsg.SteeringTorquePercent = CtrlSteeringTorquePercent;
                stcMsg.SteeringTorqueMap = SelectedTorqueMap;  
                stcMsg.SteeringControlEnabled = EnableSteeringControl;
            }
            else
            {
                stcMsg.SteeringTorquePercent = 0;
                stcMsg.SteeringTorqueMap = 0;
                stcMsg.SteeringControlEnabled = EnableSteeringControl;
            }
            VisionCmdProc.SendSteeringTorqueContronMsgOnCmdPort(stcMsg);
            return error;
        }

        public bool resetSteeringControl()
        {
            bool error = false;
            CtrlSteeringTorquePercent = 0.0;
            SelectedTorqueMap = 0;
            EnableSteeringControl = false;
            ManualControlCmd = false;
            sendSteeringTorqueControl();
            return error;
        }


        public void processSteeringStatusMsg(DceEPASteeringStatusPBMsg statusMsg)
        {
            MotorTorque = statusMsg.MotorTorquePercent;
            PWMDutyCyclePercent = statusMsg.PWMDutyCyclePercent;
            MotorCurrentAmps = statusMsg.MotorCurrentAmps;

            SupplyVoltage = statusMsg.SupplyVoltage; 
            SwitchPosition = (uint)statusMsg.SwitchPosition;
            TempDegC = statusMsg.TempDegC;
            TorqueA = (int)statusMsg.TorqueA;
            TorqueB = (int)statusMsg.TorqueB;
            SteeringAngleDeg = statusMsg.SteeringAngleDeg;
            TorqueMapActual = (uint)statusMsg.SteeringTorqueMapSetting;
            ErrorCode = (uint)statusMsg.ErrorCode;
            StatusFlags = (uint)statusMsg.StatusFlags;
            LimitFlags = (uint)statusMsg.LimitFlags;
            ManualControlFb = statusMsg.ManualExtControl;
        }

    }
}
