using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using CarCANBusMonitor;

namespace CarCANBusMonitor.Widgets
{
    public partial class LinearActuatorPositionControl : UserControl
    {

        public KarTeckLinearActuator Actuator = null;

        public UInt32 LinearActuatorTxID
        {
            get 
            {
                if (Actuator != null)
                    return Actuator.CommandID;
                else
                    return KarTeckLinearActuator.DefaultCmdID;
            }
        }

        public UInt32 LinearActuatorRxID
        {
            get 
            {
                if (Actuator != null)
                    return Actuator.ReportID;
                else
                    return KarTeckLinearActuator.DefaultRptID;
            }
        }

        private bool _scrollBarSendPosition = true;
        private bool _ignoreChange = false;
        private bool _sendPositionRpt = false;

        private byte[] _msgBuf;

        public LinearActuatorPositionControl()
        {
            InitializeComponent();
            _msgBuf = new byte[8];

            cboxModeSelect.Items.Add("Auto");
            cboxModeSelect.Items.Add("Manual");
            cboxModeSelect.Items.Add("Setup");

            _ignoreChange = true;
            cboxModeSelect.SelectedIndex = 0;
            _ignoreChange = false;
        }

        public void SetFunctionName(string name)
        {
            tbFunctionName.Text = name;
        }

        public void ClearPositionAndEnables()
        {
            tbSetPositionPercent.Text = "0";
            _scrollBarSendPosition = false;
            hScrollBarPos.Value = 0;
            _scrollBarSendPosition = true;
            Actuator.DesiredPositionPercent = 0;

            _ignoreChange = true;
            chkBxEnableClutch.Checked = false;
            Actuator.EnableClutchMc = false;

            _ignoreChange = true;
            chkBxEnableMotor.Checked = false;
            Actuator.EnableMotorMc = false;

            _ignoreChange = false;
        }

        public void ReadAndDisplayActuatorValues()
        {
            if (Actuator != null)
            {
                double posPercent = 0;
                tbSetPositionPercent.Text = Actuator.DesiredPositionPercent.ToString();
                posPercent = Actuator.DesiredPositionPercent;
                int sbPos = (int)Math.Round(posPercent);
                _scrollBarSendPosition = false;
                hScrollBarPos.Value = sbPos;
                _scrollBarSendPosition = true;

                _ignoreChange = true;
                chkBxEnableClutch.Checked = Actuator.EnableClutchMc;

                _ignoreChange = true;
                chkBxEnableMotor.Checked = Actuator.EnableMotorMc;

                _ignoreChange = true;
                if (Actuator.SetupModeCmd)
                {
                    cboxModeSelect.SelectedIndex = 2;
                }
                else if (Actuator.ManualControlCmd)
                {
                    cboxModeSelect.SelectedIndex = 1;
                }
                else
                {
                    cboxModeSelect.SelectedIndex = 0;
                }
                _ignoreChange = false;
            }
        }

        private void btnSendPos_Click(object sender, EventArgs e)
        {
            double posPercent = 0;
            if (Actuator.ManualControlCmd && Actuator != null)
            {
                double.TryParse(tbSetPositionPercent.Text, out posPercent);
                //force the scroll bar to match this position;
                Actuator.DesiredPositionPercent = posPercent;
                posPercent = Actuator.DesiredPositionPercent;
                int sbPos = (int)Math.Round(posPercent);
                _scrollBarSendPosition = false;
                hScrollBarPos.Value = sbPos;
                _scrollBarSendPosition = true;
                Actuator.sendPositionPercent();
            }
        }

        private void hScrollBarPos_Scroll(object sender, ScrollEventArgs e)
        {
            if (Actuator.ManualControlCmd && _scrollBarSendPosition && Actuator != null )
            {
                Actuator.DesiredPositionPercent = (double)hScrollBarPos.Value;
                tbSetPositionPercent.Text = Actuator.DesiredPositionPercent.ToString("0.000");
                Actuator.sendPositionPercent();
            }
        }

        private void chkBxEnableClutch_CheckedChanged(object sender, EventArgs e)
        {
            Actuator.EnableClutchMc = chkBxEnableClutch.Checked;
            if (Actuator.ManualControlCmd && !_ignoreChange )
            {
                Actuator.sendPositionPercent();
            }
            _ignoreChange = false;
        }

        private void chkBxEnableMotor_CheckedChanged(object sender, EventArgs e)
        {
            Actuator.EnableMotorMc = chkBxEnableMotor.Checked;
            if (Actuator.ManualControlCmd && !_ignoreChange )
            {
                Actuator.sendPositionPercent();
            }
            _ignoreChange = false;
        }

        public void displayPostionStatusInfo()
        {
            UInt32 val;
            tbActualPosPercent.Text = Actuator.CurrentPositionPercent.ToString("0.000");
            tbMotorCurrent.Text = Actuator.MotorCurrentAmps.ToString("0.000");
            tbMaxMotorCurrent.Text = Actuator.MaxMotorCurrentAmps.ToString("0.000");
            tbCtrlStatus.Text = Actuator.ManualControlFb ? "Manual Ctrl" : "Auto Ctrl";
            cbClutchEnFb.Checked = Actuator.ClutchEnabledFb;
            cbMotorEnFb.Checked = Actuator.MotorEnabledFb;
            Actuator.AcuatorPostionInfoChanged = false;
        }

        private void btnClearMaxMotorCurrent_Click(object sender, EventArgs e)
        {
            Actuator.ClearMaxMotorCurrent();
        }

        private void cboxModeSelect_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (!_ignoreChange)
            {
                switch (cboxModeSelect.SelectedIndex)
                {
                    case 0:
                        Actuator.SetupModeCmd = false;
                        Actuator.ManualControlCmd = false;
                        ClearPositionAndEnables();
                        break;
                    case 1:
                        Actuator.SetupModeCmd = false;
                        Actuator.ManualControlCmd = true;
                        break;
                    case 2:
                        Actuator.SetupModeCmd = true;
                        Actuator.ManualControlCmd = false;
                        ClearPositionAndEnables();
                        break;
                    default:
                        Actuator.SetupModeCmd = false;
                        Actuator.ManualControlCmd = false;
                        ClearPositionAndEnables();
                        break;
                }
                Actuator.sendPositionPercent();
            }
        }


    }
}
