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
    public partial class CameraParametersSetupWidget : UserControl
    {
        private CameraParametersSetupPBMsg _cameraParametersSetupPBMsg;

        public VisionCmdProcess VisionCmdProc;

        public CameraParametersSetupWidget()
        {
            InitializeComponent();

            cbImageFormat.Items.Add(CPImageCaptureFormat_e.Unknown);
            cbImageFormat.Items.Add(CPImageCaptureFormat_e.Grey8);
            cbImageFormat.Items.Add(CPImageCaptureFormat_e.Grey16);
            cbImageFormat.Items.Add(CPImageCaptureFormat_e.RGB24);
            cbImageFormat.Items.Add(CPImageCaptureFormat_e.BGR24);
            cbImageFormat.Items.Add(CPImageCaptureFormat_e.MJPEG);
            cbImageFormat.Items.Add(CPImageCaptureFormat_e.YUV422);
            cbImageFormat.SelectedIndex = 3;

        }

        public void ReadValues()
        {
            double dval = 0;
            int ival = 0;
            _cameraParametersSetupPBMsg.Clear();

            _cameraParametersSetupPBMsg.ImageCaptureFormat = (CPImageCaptureFormat_e)cbImageFormat.SelectedItem;

            if (int.TryParse(tbMode.Text, out ival))
            {
                _cameraParametersSetupPBMsg.Mode = ival < 0 ? 0 : (UInt32)ival;
            }
            if (int.TryParse(tbImageWidth.Text, out ival))
            {
                _cameraParametersSetupPBMsg.FrameWidth = ival < 0 ? 0 : (UInt32)ival;
            }
            if (int.TryParse(tbImageHeight.Text, out ival))
            {
                _cameraParametersSetupPBMsg.FrameHeight = ival < 0 ? 0 : (UInt32)ival;
            }

        }


        private void btSendParameters_Click(object sender, EventArgs e)
        {
            ReadValues();
            if (VisionCmdProc != null)
            {
                VisionCmdProc.SendCameraParameersMsg(_cameraParametersSetupPBMsg);
            }
        }
    }
}
