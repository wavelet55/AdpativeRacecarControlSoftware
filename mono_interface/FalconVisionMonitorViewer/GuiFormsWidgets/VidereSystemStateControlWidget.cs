using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using VisionBridge.Messages;

namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    public partial class VidereSystemStateControlWidget : UserControl
    {

        //Videre System States that can be sent... See VidereSystemStates_e
        public enum VidereSystemStates_e
        {
            Initialize,
            RemoteControl,
            ExternalMonitorControl,   //Manual Control
            ManualDriverControl,
            HeadOrientedCalibration,
            HeadOrientedControl,
        }

        public VisionCmdProcess VisionCmdProc = null;

        public VidereSystemStates_e VidereSystemCurrentState = VidereSystemStates_e.Initialize;

        public Action ResetAllVehicleActuators = null;

        private VidereSystemStates_e _lastSystemStateCmd;

        private bool _ignoreChange = false;

        public VidereSystemStateControlWidget()
        {
            InitializeComponent();

            //Setup Videre System State Control
            cbSetSystemState.Items.Add("Manual Control");
            cbSetSystemState.Items.Add("Remote Control");
            cbSetSystemState.Items.Add("Calibration");
            cbSetSystemState.Items.Add("Safety Driver");
            cbSetSystemState.Items.Add("Head Control");

            _ignoreChange = true;
            cbSetSystemState.SelectedIndex = 4;
            _lastSystemStateCmd = VidereSystemStates_e.HeadOrientedControl;

            _ignoreChange = true;
            chkBxHeadEnable.Checked = true;
            _ignoreChange = true;
            chkBxThrottleEnable.Checked = true;
            _ignoreChange = true;
            chkBxBrakeEnable.Checked = true;

            _ignoreChange = true;
            ckboxBCI_En_Ctrl.Checked = false;
        }

        private void readAndSendControlCmdInfo()
        {
            VidereSystemControlPBMsg ctrlMsg = new VidereSystemControlPBMsg();

            VidereSystemStates_e systemStateCmd = VidereSystemStates_e.ExternalMonitorControl;
            switch (cbSetSystemState.SelectedIndex)
            {
                case 0:     //"Manual Control"
                    systemStateCmd = VidereSystemStates_e.ExternalMonitorControl;
                    break;

                case 1:     //"Remote Control"
                    systemStateCmd = VidereSystemStates_e.RemoteControl;
                    break;

                case 2:     //"Calibration"
                    systemStateCmd = VidereSystemStates_e.HeadOrientedCalibration;
                    break;

                case 3:     //"Safety Driver"
                    systemStateCmd = VidereSystemStates_e.ManualDriverControl;
                    break;

                case 4:     //"Head Control"
                    systemStateCmd = VidereSystemStates_e.HeadOrientedControl;
                    break;
            }

            if (VisionCmdProc != null)
            {
                ctrlMsg.SystemState = (UInt32)systemStateCmd;
                bool StateCmdChanged = systemStateCmd != _lastSystemStateCmd;
                _lastSystemStateCmd = systemStateCmd;
                ctrlMsg.HeadControlEnable = chkBxHeadEnable.Checked;
                ctrlMsg.ThrottleControlEnable = chkBxThrottleEnable.Checked;
                ctrlMsg.BrakeControlEnable = chkBxBrakeEnable.Checked;
                ctrlMsg.NexusBCIControlEnabled = ckboxBCI_En_Ctrl.Checked;

                VisionCmdProc.SendVidereSystemControlMsg(ctrlMsg);
                if (ResetAllVehicleActuators != null && StateCmdChanged)
                {
                    ResetAllVehicleActuators();
                }
            }


        }

        private void cbSetSystemState_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (!_ignoreChange)
            {
                readAndSendControlCmdInfo();
            }
            _ignoreChange = false;
        }

        public void ProcessVidereSystemStateMsg(VidereSystemControlPBMsg stateMsg)
        {
            VidereSystemCurrentState = (VidereSystemStates_e)stateMsg.SystemState;
            switch (VidereSystemCurrentState)
            {
                case VidereSystemStates_e.Initialize:
                    tbSystemState.Text = "Initialize";
                    break;

                case VidereSystemStates_e.ExternalMonitorControl:
                    tbSystemState.Text = "Manual Control";
                    break;
                case VidereSystemStates_e.RemoteControl:
                    tbSystemState.Text = "Remote Control";
                    break;
                case VidereSystemStates_e.HeadOrientedCalibration:
                    tbSystemState.Text = "Calibration";
                    break;
                case VidereSystemStates_e.ManualDriverControl:
                    tbSystemState.Text = "Safety Driver";
                    break;
                case VidereSystemStates_e.HeadOrientedControl:
                    tbSystemState.Text = "Head Control";
                    break;
            }

            chkBxDriverEnableSwitch.Checked = stateMsg.DriverEnableSwitch;
            chkBxHeadControlFB.Checked = stateMsg.HeadControlEnable;
            chkBxThrottleControlFB.Checked = stateMsg.ThrottleControlEnable;
            chkBxBrakeControlFB.Checked = stateMsg.BrakeControlEnable;
            ckBox_BCI_En_State.Checked = stateMsg.NexusBCIControlEnabled;

            if(stateMsg.NexusBCIControlEnabled)
            {
                if(stateMsg.NexusBCIThrottleEnable)
                {
                    textBoxBCI_ThottleCtrlState.Text = "BCI Throttle ON";
                    textBoxBCI_ThottleCtrlState.BackColor = System.Drawing.Color.Green;
                }
                else
                {
                    textBoxBCI_ThottleCtrlState.Text = "BCI OFF";
                    textBoxBCI_ThottleCtrlState.BackColor = System.Drawing.Color.LightBlue;
                }
            }
            else
            {
                textBoxBCI_ThottleCtrlState.Text = "BCI Not Enabled";
                textBoxBCI_ThottleCtrlState.BackColor = System.Drawing.Color.Yellow;
            }
        }

        private void chkBxHeadEnable_CheckedChanged(object sender, EventArgs e)
        {
            if (!_ignoreChange)
            {
                readAndSendControlCmdInfo();
            }
            _ignoreChange = false;
        }

        private void chkBxThrottleEnable_CheckedChanged(object sender, EventArgs e)
        {
            if (!_ignoreChange)
            {
                readAndSendControlCmdInfo();
            }
            _ignoreChange = false;
        }

        private void chkBxBrakeEnable_CheckedChanged(object sender, EventArgs e)
        {
            if (!_ignoreChange)
            {
                readAndSendControlCmdInfo();
            }
            _ignoreChange = false;
        }

        private void ckboxBCI_En_Ctrl_CheckedChanged(object sender, EventArgs e)
        {
            if (!_ignoreChange)
            {
                readAndSendControlCmdInfo();
            }
            _ignoreChange = false;
        }
    }
}
