using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CarCANBusMonitor.Widgets
{
    public partial class SteeringControlWidget : UserControl
    {
        public DceEPAS EPAS_Obj = null;

        double SteeringTorquePercent;

        private bool _ignoreChange = false;

        public SteeringControlWidget()
        {
            InitializeComponent();

            cbTorqueMapSetting.Items.Add(0);
            cbTorqueMapSetting.Items.Add(1);
            cbTorqueMapSetting.Items.Add(2);
            cbTorqueMapSetting.Items.Add(3);
            cbTorqueMapSetting.Items.Add(4);
            cbTorqueMapSetting.Items.Add(5);
            cbTorqueMapSetting.SelectedIndex = 3;
        }

        public void ReadAndDisplaySteeringControlValues()
        {
            if (EPAS_Obj != null)
            {
                _ignoreChange = true;
                cbTorqueMapSetting.SelectedIndex = (int)EPAS_Obj.SelectedTorqueMap;

                _ignoreChange = true;
                cbManualCtrlEnable.Checked = EPAS_Obj.ManualControlCmd;

                _ignoreChange = true;
                chkBxEnableSteeringCtrl.Enabled = EPAS_Obj.EnableSteeringControl;

                _ignoreChange = true;
                hScrollBarPos.Value = (int)EPAS_Obj.CtrlSteeringTorquePercent;

                _ignoreChange = false;
            }
        }

        private void cbTorqueMapSetting_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (EPAS_Obj != null)
            {
                EPAS_Obj.SelectedTorqueMap = (UInt32)cbTorqueMapSetting.SelectedIndex;
                if(cbManualCtrlEnable.Checked  && !_ignoreChange)
                {
                    EPAS_Obj.sendSteeringTorqueControl();
                }
                _ignoreChange = false;
            }
        }

        private void hScrollBarPos_Scroll(object sender, ScrollEventArgs e)
        {
            if (EPAS_Obj != null)
            {
                EPAS_Obj.CtrlSteeringTorquePercent = (double)hScrollBarPos.Value;
                if(cbManualCtrlEnable.Checked   && !_ignoreChange)
                {
                    EPAS_Obj.sendSteeringTorqueControl();
                }
                _ignoreChange = false;
            }
        }

        public void displaySteeringStatusInfo()
        {
            tbMotorTorque.Text = EPAS_Obj.MotorTorque.ToString("0.0");
            tbMotorPWM.Text = EPAS_Obj.PWMDutyCyclePercent.ToString("0.0");
            tbMotorCurrent.Text = EPAS_Obj.MotorCurrentAmps.ToString("0.0");
            tbMotorVolts.Text = EPAS_Obj.SupplyVoltage.ToString("0.0");
            tbCtlrBoxTempC.Text = EPAS_Obj.TempDegC.ToString("0.0");
            tbTorqueA.Text = EPAS_Obj.TorqueA.ToString();
            tbTorqueB.Text = EPAS_Obj.TorqueB.ToString();
            tbSteeringAngleDeg.Text = EPAS_Obj.SteeringAngleDeg.ToString("0.0");
            tbTorqueMapActual.Text = EPAS_Obj.TorqueMapActual.ToString();
            tbErrorCode.Text = EPAS_Obj.ErrorCode.ToString();
            chkBxMoveLeft.Checked = EPAS_Obj.IsMotorMovingLeft;
            chkBxMoveRight.Checked = EPAS_Obj.IsMotorMovingRight;
            chkBxFault.Checked = EPAS_Obj.IsFaultLight;
        }

        private void cbManualCtrlEnable_CheckedChanged(object sender, EventArgs e)
        {
            if (EPAS_Obj != null)
            {
                EPAS_Obj.ManualControlCmd = cbManualCtrlEnable.Checked;
                if (!_ignoreChange)
                {
                    EPAS_Obj.sendSteeringTorqueControl();
                }
                _ignoreChange = false;
            }
        }

        private void chkBxEnableSteeringCtrl_CheckedChanged(object sender, EventArgs e)
        {
            if (EPAS_Obj != null)
            {
                EPAS_Obj.EnableSteeringControl = chkBxEnableSteeringCtrl.Enabled;
                if (!_ignoreChange && cbManualCtrlEnable.Checked)
                {
                    EPAS_Obj.sendSteeringTorqueControl();
                }
                _ignoreChange = false;
            }
        }

        private void timerSendSteeringPos_Tick(object sender, EventArgs e)
        {
            if (EPAS_Obj != null && cbManualCtrlEnable.Checked)
            {
                //EPAS_Obj.sendSteeringTorqueControl();
            }
        }

    }
}
