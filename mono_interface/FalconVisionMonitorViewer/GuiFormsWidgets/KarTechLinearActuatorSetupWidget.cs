using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows;

namespace CarCANBusMonitor.Widgets
{
    public partial class KarTechLinearActuatorSetupWidget : UserControl
    {
        public KarTeckLinearActuator LinearActuator = null;

        public KarTechLinearActuatorSetupWidget()
        {
            InitializeComponent();
        }

        private void groupBoxLASetup_Enter(object sender, EventArgs e)
        {

        }

        private void btnReadConfigParameters_Click(object sender, EventArgs e)
        {
            if (LinearActuator != null)
            {
                if (LinearActuator.getKarTeckActuatorConfigParams())
                {
                    UpdateParameters();
                }
            }
        }

        private void btnSetConfigParameters_Click(object sender, EventArgs e)
        {
            if (LinearActuator != null)
            {
		        string message = "Are you sure you want to do this?";
		        string caption = "Set All Actuator Parameters";
		        MessageBoxButtons buttons = MessageBoxButtons.YesNo;
		        DialogResult result;
                result = MessageBox.Show(this, message, caption, buttons);
                if (result == DialogResult.Yes)
                {
                    UInt32 uiVal = 0;
                    double dVal = 0;
                    bool parseOk = true;

                    //Don't allow changing at this time.
                    //if (double.TryParse(tbMotorMaxCurrentSetting.Text, out dVal))
                    //{
                    //    LinearActuator.MotorMaxCurrentLimitAmps = dVal;
                    //}

                    if( double.TryParse(tbMinPosition.Text, out dVal) )
                        LinearActuator.MinPositionInches = dVal;

                    if( double.TryParse(tbMaxPosition.Text, out dVal) )
                        LinearActuator.MaxPositionInches = dVal;

                    if (UInt32.TryParse(tbPosReachedErrTime.Text, out uiVal))
                    {
                        LinearActuator.PositionReachedErrorTimeMSec = uiVal;
                    }

                    if( UInt32.TryParse(tbFeedbackLoop_KP.Text, out uiVal) )
                        LinearActuator.FeedbackCtrl_KP = uiVal;

                    if( UInt32.TryParse(tbFeedbackLoop_KI.Text, out uiVal) )
                        LinearActuator.FeedbackCtrl_KI = uiVal;

                    if( UInt32.TryParse(tbFeedbackLoop_KD.Text, out uiVal) )
                        LinearActuator.FeedbackCtrl_KD = uiVal;

                    if( UInt32.TryParse(tbFeedbackLoopCLFreq.Text, out uiVal) )
                        LinearActuator.FeedbackCtrl_CLFreq = uiVal;

                    if( double.TryParse(tbFeedbackLoopDeadband.Text, out dVal) )
                        LinearActuator.FeedbackCtrl_ErrDeadbandInches = dVal;

                    if( UInt32.TryParse(tbMotorPWMFreq.Text, out uiVal) )
                        LinearActuator.Motor_pwmFreq = uiVal;

                    if( UInt32.TryParse(tbMotorMinPwmPercent.Text, out uiVal) )
                        LinearActuator.Motor_MinPWM = uiVal;

                    if( UInt32.TryParse(tbMotorMaxPwmPercent.Text, out uiVal) )
                        LinearActuator.Motor_MaxPWM = uiVal;

                    if (parseOk)
                    {
                        LinearActuator.sendKarTeckActuatorConfigParams();
                    }
                }
            }
        }

        private void btnResetConfigs_Click(object sender, EventArgs e)
        {
            if (LinearActuator != null)
            {
		        string message = "Are you sure you want to do this?";
		        string caption = "Reset Actuator Parameters";
		        MessageBoxButtons buttons = MessageBoxButtons.YesNo;
		        DialogResult result;
                result = MessageBox.Show(this, message, caption, buttons);
                if (result == DialogResult.Yes)
                {
                    LinearActuator.SetFactoryDefaultValues();

                    UInt32 resetFlags = 0;
                    resetFlags |= 0x0008;   //Reset Hardware Configs.
                    resetFlags |= 0x0010;   //Reset User Configs (KP, KI, KD, ...)
                    //LinearActuator.resetActuator(resetFlags, 0);
                }
            }
        }


        public void UpdateParameters()
        {
            if (LinearActuator != null)
            {
                tbActuatorFunction.Text = LinearActuator.LinearActuatorFunction.ToString();

                tbMinPosition.Text = LinearActuator.MinPositionInches.ToString("0.000");
                tbMaxPosition.Text = LinearActuator.MaxPositionInches.ToString("0.000");

                tbMotorMaxCurrentSetting.Text = LinearActuator.MotorMaxCurrentLimitAmps.ToString("0.000");
                tbPosReachedErrTime.Text = LinearActuator.PositionReachedErrorTimeMSec.ToString();

                tbFeedbackLoop_KP.Text = LinearActuator.FeedbackCtrl_KP.ToString();
                tbFeedbackLoop_KI.Text = LinearActuator.FeedbackCtrl_KI.ToString();
                tbFeedbackLoop_KD.Text = LinearActuator.FeedbackCtrl_KD.ToString();
                tbFeedbackLoopCLFreq.Text = LinearActuator.FeedbackCtrl_CLFreq.ToString();
                tbFeedbackLoopDeadband.Text = LinearActuator.FeedbackCtrl_ErrDeadbandInches.ToString("0.000");

                tbMotorPWMFreq.Text = LinearActuator.Motor_pwmFreq.ToString();
                tbMotorMinPwmPercent.Text = LinearActuator.Motor_MinPWM.ToString();
                tbMotorMaxPwmPercent.Text = LinearActuator.Motor_MaxPWM.ToString();
            }
        }

        private void btnAutoZeroPosSensor_Click(object sender, EventArgs e)
        {
            if (LinearActuator != null)
            {
		        string message = "Only Zero the sensor when the actuator shaft is free to move over its full range. Are you sure you want to do this?";
		        string caption = "Zero Position Sensor";
		        MessageBoxButtons buttons = MessageBoxButtons.YesNo;
		        DialogResult result;
                result = MessageBox.Show(this, message, caption, buttons);
                if (result == DialogResult.Yes)
                {
                    LinearActuator.sendAutoZeroCalCommand();
                }
            }
        }

        private void btnResetOutput_Click(object sender, EventArgs e)
        {
            if (LinearActuator != null)
            {
		        string message = "Are you sure you want to reset the output?";
		        string caption = "Reset Actuator Output";
		        MessageBoxButtons buttons = MessageBoxButtons.YesNo;
		        DialogResult result;
                result = MessageBox.Show(this, message, caption, buttons);
                if (result == DialogResult.Yes)
                {
                    LinearActuator.sendOutputResetCommand();
                }
            }
        }

        private void btnResetHdwrCfgs_Click(object sender, EventArgs e)
        {
            if (LinearActuator != null)
            {
		        string message = "Are you sure you want to reset the hardware configs?";
		        string caption = "Reset Actuator Hardware Configs";
		        MessageBoxButtons buttons = MessageBoxButtons.YesNo;
		        DialogResult result;
                result = MessageBox.Show(this, message, caption, buttons);
                if (result == DialogResult.Yes)
                {
                    LinearActuator.sendHardwareConfigResetCommand();
                }
            }
        }

        private void btnResetUserConfigs_Click(object sender, EventArgs e)
        {
            if (LinearActuator != null)
            {
		        string message = "Are you sure you want to reset the user configs?";
		        string caption = "Reset Actuator User Configs";
		        MessageBoxButtons buttons = MessageBoxButtons.YesNo;
		        DialogResult result;
                result = MessageBox.Show(this, message, caption, buttons);
                if (result == DialogResult.Yes)
                {
                    LinearActuator.sendUserConfigResetCommand();
                }
            }
        }

        private void btnResetAllCfgs_Click(object sender, EventArgs e)
        {
            if (LinearActuator != null)
            {
		        string message = "Are you sure you want to reset All the configs?  This will require reseting up the KarTech Command and Report IDs/Addresses. Don't do this unless you are really-really sure!";
		        string caption = "Reset All the Actuator Configurations";
		        MessageBoxButtons buttons = MessageBoxButtons.YesNo;
		        DialogResult result;
                result = MessageBox.Show(this, message, caption, buttons);
                if (result == DialogResult.Yes)
                {
                    LinearActuator.sendResetAllCommand();
                }
            }
        }

        private void btnSetCmdRptIDs_Click(object sender, EventArgs e)
        {
            if (LinearActuator != null)
            {
		        string message = "Are you sure you want to set the CAN Bus IDs?  Only one KarTech Actuator can be attached to the CAN bus at a time!!!";
		        string caption = "Set the CAN bus Comand and Response IDs";
		        MessageBoxButtons buttons = MessageBoxButtons.YesNo;
		        DialogResult result;
                result = MessageBox.Show(this, message, caption, buttons);
                if (result == DialogResult.Yes)
                {
                    LinearActuator.sendSetComandResponseIDsCommand();
                }
            }
        }
    }
}
