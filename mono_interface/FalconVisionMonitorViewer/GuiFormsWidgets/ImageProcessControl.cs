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
    public partial class ImageProcessControl : UserControl
    {
        private VisionProcessingControlMsg _visionProcessingControlMsg;

        private bool _ignoreTDTChange = false;

        public VisionCmdProcess VisionCmdProc;

        public bool TargetProcessingEnabled = false;

        public ImageProcessControl()
        {
            InitializeComponent();

            cbVisionProcessorType.Items.Add(VisionProcessingMode_e.VPM_None);
            cbVisionProcessorType.Items.Add(VisionProcessingMode_e.VPM_Target);
            cbVisionProcessorType.Items.Add(VisionProcessingMode_e.VPM_GPSDenied);
            cbVisionProcessorType.Items.Add(VisionProcessingMode_e.VPM_CameraCalibration);
            cbVisionProcessorType.Items.Add(VisionProcessingMode_e.VPM_FeatureMatchProc);
            cbVisionProcessorType.Items.Add(VisionProcessingMode_e.VPM_HeadTrackingProc);
            cbVisionProcessorType.Items.Add(VisionProcessingMode_e.VPM_HeadOrientationCalProc);
            cbVisionProcessorType.SelectedItem = VisionProcessingMode_e.VPM_HeadTrackingProc;

            cbTgtDetectionType.Items.Add(TargetProcessingMode_e.TgtProcMode_None);
            cbTgtDetectionType.Items.Add(TargetProcessingMode_e.TgtProcMode_Std);
            cbTgtDetectionType.Items.Add(TargetProcessingMode_e.TgtProcMode_Blob);
            cbTgtDetectionType.Items.Add(TargetProcessingMode_e.TgtProcMode_CheckerBoard);
            cbTgtDetectionType.SelectedItem = TargetProcessingMode_e.TgtProcMode_Std;

            //Use GPU Acceleration if available.
            chkBxGPUTgtDetectionEnabled.Checked = true;

            _visionProcessingControlMsg = new VisionProcessingControlMsg();

            btnImgProcEnabled.Text = "Disabled";
            btnImgProcEnabled.BackColor = System.Drawing.Color.LightGray;
        }

        private void btnSendImgProcCtrlMsg_Click(object sender, EventArgs e)
        {
            _visionProcessingControlMsg.Clear();
            _visionProcessingControlMsg.VisionProcessingMode = (VisionProcessingMode_e)cbVisionProcessorType.SelectedItem;
            _visionProcessingControlMsg.TargetImageProcessingEnabled = TargetProcessingEnabled;
            _visionProcessingControlMsg.TargetProcessingMode = (TargetProcessingMode_e)cbTgtDetectionType.SelectedItem;
            _visionProcessingControlMsg.GPUProcessingEnabled = chkBxGPUTgtDetectionEnabled.Checked;
            VisionCmdProc.SendVisionProcessControlCmd(_visionProcessingControlMsg);
        }

        private void cbTgtDetectionType_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (!_ignoreTDTChange)
            {

            }
            _ignoreTDTChange = false;
        }

        public void SetImageProcessCmdStatus(VisionProcessingControlMsg visionProcessCtrlStatusMsg)
        {
            if (visionProcessCtrlStatusMsg != null)
            {
                tbVisionProcModeSelected.Text = visionProcessCtrlStatusMsg.VisionProcessingMode.ToString();

                if (visionProcessCtrlStatusMsg.TargetImageProcessingEnabled)
                {
                    btnImgProcEnabled.Text = "Enabled";
                    btnImgProcEnabled.BackColor = System.Drawing.Color.Green;
                }
                else
                {
                    btnImgProcEnabled.Text = "Disabled";
                    btnImgProcEnabled.BackColor = System.Drawing.Color.LightSalmon;
                }

                tbActualTgtDetectType.Text = visionProcessCtrlStatusMsg.TargetProcessingMode.ToString();

                if (visionProcessCtrlStatusMsg.GPUProcessingEnabled)
                {
                    tbGPUStatus.Text = "Yes";
                }
                else
                {
                    tbGPUStatus.Text = "No";
                }
            }
        }
    }
}
