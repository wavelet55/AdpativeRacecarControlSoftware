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
    public partial class VehicleControlParametersWidget : UserControl
    {

        public VisionCmdProcess VisionCmdProc = null;

        public VehicleControlParametersPBMsg VehicleControlParametersMsg;


        public VehicleControlParametersWidget()
        {
            InitializeComponent();
            VehicleControlParametersMsg = new VehicleControlParametersPBMsg();
            setParamters(VehicleControlParametersMsg);

            cmbBxLPFOrder.Items.Add(0);
            cmbBxLPFOrder.Items.Add(2);
            cmbBxLPFOrder.Items.Add(4);
            cmbBxLPFOrder.Items.Add(6);
            cmbBxLPFOrder.SelectedIndex = 1;
        }

        public void setParamters(VehicleControlParametersPBMsg vcParamsMsg)
        {
            tbSipnPuffDeadband.Text = vcParamsMsg.SipnPuffDeadBandPercent.ToString("0.000");
            tbSipnPuffBlowGain.Text = vcParamsMsg.SipnPuffBlowGain.ToString("0.000");
            tbSipnPuffSuckGain.Text = vcParamsMsg.SipnPuffSuckGain.ToString("0.000");
            tbBCI_Gain.Text = vcParamsMsg.BCIGain.ToString("0.000");

            tbMaxHeadLRRotation.Text = vcParamsMsg.MaxLRHeadRotationDegrees.ToString("0.0");
            tbSteeringDeadBand.Text = vcParamsMsg.SteeringDeadband.ToString("0.000");
            tbSteeringGain.Text = vcParamsMsg.SteeringControlGain.ToString("0.000");
            tbSteeringBiasAngle.Text = vcParamsMsg.SteeringBiasAngleDegrees.ToString("0.000");
            tbRCSteeringGain.Text = vcParamsMsg.RCSteeringGain.ToString("0.000");

            chkBxSteeringAngleTorqueSelect.Checked = vcParamsMsg.UseSteeringAngleControl;
            int lpfo = vcParamsMsg.HeadLeftRighLPFOrder;
            lpfo = lpfo < 0 ? 0 : lpfo > 6 ? 6 : lpfo;
            cmbBxLPFOrder.SelectedItem = lpfo / 2;

            tbSteeringLPFCutoffFreq.Text = vcParamsMsg.HeadLeftRighLPFCutoffFreqHz.ToString("0.000");
            
            tbSteeringFB_Kp.Text = vcParamsMsg.SteeringAngleFeedback_Kp.ToString("0.000");
            tbSteeringFB_Kd.Text = vcParamsMsg.SteeringAngleFeedback_Kd.ToString("0.000");
            tbSteeringFB_Ki.Text = vcParamsMsg.SteeringAngleFeedback_Ki.ToString("0.000");
        }


        public void readParameters()
        {
            double val;
            if (!double.TryParse(tbSipnPuffDeadband.Text, out val))
            {
                val = 2.5;
                tbSipnPuffDeadband.Text = val.ToString();
            }
            val = val < 0 ? 0 : val > 25.0 ? 25.0 : val;
            VehicleControlParametersMsg.SipnPuffDeadBandPercent = val;

            if (!double.TryParse(tbSipnPuffBlowGain.Text, out val))
            {
                val = 1.0;
                tbSipnPuffBlowGain.Text = val.ToString();
            }
            val = val < 0 ? 0 : val > 10.0 ? 10.0 : val;
            VehicleControlParametersMsg.SipnPuffBlowGain = val;

            if (!double.TryParse(tbSipnPuffSuckGain.Text, out val))
            {
                val = 1.0;
                tbSipnPuffSuckGain.Text = val.ToString();
            }
            val = val < 0 ? 0 : val > 10.0 ? 10.0 : val;
            VehicleControlParametersMsg.SipnPuffSuckGain = val;

            if (!double.TryParse(tbBCI_Gain.Text, out val))
            {
                val = 1.0;
                tbBCI_Gain.Text = val.ToString();
            }
            val = val < 0 ? 0 : val > 100.0 ? 100.0 : val;
            VehicleControlParametersMsg.BCIGain = val;

            if (!double.TryParse(tbMaxHeadLRRotation.Text, out val))
            {
                val = 60.0;
                tbMaxHeadLRRotation.Text = val.ToString();
            }
            val = val < 30.0 ? 30.0 : val > 75.0 ? 75.0 : val;
            VehicleControlParametersMsg.MaxLRHeadRotationDegrees = val;

            if (!double.TryParse(tbSteeringDeadBand.Text, out val))
            {
                val = 2.5;
                tbSteeringDeadBand.Text = val.ToString();
            }
            val = val < 0 ? 0 : val > 20.0 ? 20.0 : val;
            VehicleControlParametersMsg.SteeringDeadband = val;

            if (!double.TryParse(tbSteeringGain.Text, out val))
            {
                val = 1.0;
                tbSteeringGain.Text = val.ToString();
            }
            val = val < 0.001 ? 0.001 : val > 100.0 ? 100.0 : val;
            VehicleControlParametersMsg.SteeringControlGain = val;

            if (!double.TryParse(tbSteeringBiasAngle.Text, out val))
            {
                val = 1.0;
                tbSteeringBiasAngle.Text = val.ToString();
            }
            val = val < -20.0 ? -20.0 : val > 20.0 ? 20.0 : val;
            VehicleControlParametersMsg.SteeringBiasAngleDegrees = val;

            if (!double.TryParse(tbRCSteeringGain.Text, out val))
            {
                val = 1.0;
                tbRCSteeringGain.Text = val.ToString();
            }
            val = val < 0.001 ? 0.001 : val > 100.0 ? 100.0 : val;
            VehicleControlParametersMsg.RCSteeringGain = val;


            VehicleControlParametersMsg.UseSteeringAngleControl = chkBxSteeringAngleTorqueSelect.Checked;

            VehicleControlParametersMsg.HeadLeftRighLPFOrder =  2 * (int)cmbBxLPFOrder.SelectedIndex;

            if (!double.TryParse(tbSteeringLPFCutoffFreq.Text, out val))
            {
                val = 5.0;
                tbSteeringLPFCutoffFreq.Text = val.ToString();
            }
            val = val < 1.0 ? 1.0 : val > 20.0 ? 20.0 : val;
            VehicleControlParametersMsg.HeadLeftRighLPFCutoffFreqHz = val;

            if (!double.TryParse(tbSteeringFB_Kp.Text, out val))
            {
                val = 1.0;
                tbSteeringFB_Kp.Text = val.ToString();
            }
            val = val < 0.001 ? 0.001 : val > 10000.0 ? 10000.0 : val;
            VehicleControlParametersMsg.SteeringAngleFeedback_Kp = val;

            if (!double.TryParse(tbSteeringFB_Kd.Text, out val))
            {
                val = 0.0;
                tbSteeringFB_Kd.Text = val.ToString();
            }
            val = val < 0 ? 0 : val > 1000.0 ? 1000.0 : val;
            VehicleControlParametersMsg.SteeringAngleFeedback_Kd = val;

            if (!double.TryParse(tbSteeringFB_Ki.Text, out val))
            {
                val = 0.0;
                tbSteeringFB_Ki.Text = val.ToString();
            }
            val = val < 0 ? 0 : val > 1000.0 ? 1000.0 : val;
            VehicleControlParametersMsg.SteeringAngleFeedback_Ki = val;

        }


        private void btnSendParameters_Click(object sender, EventArgs e)
        {
            readParameters();
            if (VisionCmdProc != null)
            {
                VisionCmdProc.SendVehicleControlParametersMsg(VehicleControlParametersMsg);
            }
        }


        public bool GetVehicleControlParameters()
        {
            bool paramtersReceived = false;
            if (VisionCmdProc != null)
            {
                VehicleControlParametersPBMsg vcpMsg = VisionCmdProc.GetVehicleControlParameters();
                if (vcpMsg != null)
                {
                    setParamters(vcpMsg);
                    readParameters();
                    paramtersReceived = true;
                }
            }
            return paramtersReceived;
        }


        private void btnGetParameters_Click(object sender, EventArgs e)
        {
            GetVehicleControlParameters();
        }
    }
}
