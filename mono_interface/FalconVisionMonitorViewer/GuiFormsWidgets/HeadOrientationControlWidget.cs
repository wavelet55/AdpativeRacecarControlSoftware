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
    public partial class HeadOrientationControlWidget : UserControl
    {

        public VisionCmdProcess VisionCmdProc;

        private HeadOrientationControlPBMsg HeadOrientationControlMsg;

        bool ignorechange = false;

        public HeadOrientationControlWidget()
        {
            InitializeComponent();

            HeadOrientationControlMsg = new HeadOrientationControlPBMsg();

            cbxOrientationTypeOutpSelect.Items.Add("No Output");
            cbxOrientationTypeOutpSelect.Items.Add("ImageP Head");
            cbxOrientationTypeOutpSelect.Items.Add("Head Orient");
            cbxOrientationTypeOutpSelect.Items.Add("Car Orient");
            ignorechange = true;
            cbxOrientationTypeOutpSelect.SelectedIndex = 0;
        }

        public void ReadVals()
        {
            double val = 0;
            if (!double.TryParse(tbHeadQvar.Text, out val))
            {
                val = 0.001;
                tbHeadQvar.Text = val.ToString();
            }
            HeadOrientationControlMsg.HeadOrientation_QVar = val;

            if (!double.TryParse(tbHeadRvar.Text, out val))
            {
                val = 0.001;
                tbHeadRvar.Text = val.ToString();
            }
            HeadOrientationControlMsg.HeadOrientation_RVar = val;

            if (!double.TryParse(tbGravityFBGain.Text, out val))
            {
                val = 0.99;
                tbGravityFBGain.Text = val.ToString();
            }
            HeadOrientationControlMsg.VehicleGravityFeedbackGain = val;

            HeadOrientationControlMsg.HeadOrientationSelect = (UInt32)cbxOrientationTypeOutpSelect.SelectedIndex;

            HeadOrientationControlMsg.DisableHeadOrientationKalmanFilter = ckbDisableHeadKalmanFilter.Checked;
            HeadOrientationControlMsg.DisableVehicleInputToHeadOrientation = ckbDisableCarOrInp.Checked;
            HeadOrientationControlMsg.DisableVehicleGravityFeedback = ckbDisableCarGravityFB.Checked;
        }

        private void btnSend_Click(object sender, EventArgs e)
        {
            ReadVals();
            VisionCmdProc.SendHeadOrientationCommandMsg(HeadOrientationControlMsg);
        }

        private void cbxOrientationTypeOutpSelect_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (!ignorechange)
            {
                ReadVals();
                VisionCmdProc.SendHeadOrientationCommandMsg(HeadOrientationControlMsg);
            }
            ignorechange = false;
        }

        private void ckbDisableHeadKalmanFilter_CheckedChanged(object sender, EventArgs e)
        {
            ReadVals();
            VisionCmdProc.SendHeadOrientationCommandMsg(HeadOrientationControlMsg);
        }

        private void ckbDisableCarOrInp_CheckedChanged(object sender, EventArgs e)
        {
            ReadVals();
            VisionCmdProc.SendHeadOrientationCommandMsg(HeadOrientationControlMsg);
        }

        private void ckbDisableCarGravityFB_CheckedChanged(object sender, EventArgs e)
        {
            ReadVals();
            VisionCmdProc.SendHeadOrientationCommandMsg(HeadOrientationControlMsg);
        }


    }
}
