using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using VisionBridge.Messages;

namespace FalconVisionMonitorViewer.GuiFormsWidgets
{

    public partial class ImageCaptureControl : UserControl
    {
        public ImageCaptureSource_e ImageCaptureSource = ImageCaptureSource_e.NoChange;

        private bool _imageCaptureEnabledCmd = false;

        private string openCVWebCamDevice = "0";

        private string IPM_Directory = "ImagePlusMetadata";

        private string CImgDirectory = "Images";
        private string CImgFileExt = "jpg";

        public VisionCmdProcess VisionCmdProc;


        public bool VSImageCaptureEnabled = false;

        public bool VSImageCaptureComplete = false;

        public bool VSEndOfImages = false;

        public ImageCaptureSource_e VSImageCaptureSource = ImageCaptureSource_e.NoChange;

        public ImageCaptureError_e VSImageCaptureError = ImageCaptureError_e.None;


        private ImageCaptureControlMsg _imageCaptureControlMsg;

        private bool _ignoreSChange = false;


        public ImageCaptureControl()
        {
            InitializeComponent();
            nbxNumerOfImgesToCapture.Minimum = 0;
            nbxNumerOfImgesToCapture.Value = 0;

            _imageCaptureControlMsg = new ImageCaptureControlMsg();
        }


        private void btnSingleImageCapture_Click(object sender, EventArgs e)
        {
            _imageCaptureControlMsg.Clear();
            if (!VSImageCaptureEnabled || VSImageCaptureComplete)
            {
                //First Disable 
                _imageCaptureEnabledCmd = false;
                _imageCaptureControlMsg.Clear();
                _imageCaptureControlMsg.ImageCaptureEnabled = false;
                VisionCmdProc.SendImageCaptureControlCmd(_imageCaptureControlMsg);

                Thread.Sleep(100);

                _imageCaptureControlMsg.Clear();
                _imageCaptureControlMsg.ImageCaptureEnabled = true;
                _imageCaptureControlMsg.NumberOfImagesToCapture = 1;  //Process 1 Image
                _imageCaptureControlMsg.ImageSourceLoopAround = cbContinuousCapture.Checked;
                VisionCmdProc.SendImageCaptureControlCmd(_imageCaptureControlMsg);
                _imageCaptureEnabledCmd = true;
                btEnableImageCapture.Text = "Disable";
                btEnableImageCapture.BackColor = System.Drawing.Color.Green; 
            }

        }

        private void btEnableImageCapture_Click(object sender, EventArgs e)
        {
            if (_imageCaptureEnabledCmd || VSImageCaptureEnabled)
            {
                //Send a disable command
                _imageCaptureEnabledCmd = false;
                _imageCaptureControlMsg.Clear();
                _imageCaptureControlMsg.ImageCaptureEnabled = false;
                _imageCaptureControlMsg.NumberOfImagesToCapture = 0;
                VisionCmdProc.SendImageCaptureControlCmd(_imageCaptureControlMsg);
                btEnableImageCapture.Text = "Enable";
                btEnableImageCapture.BackColor = System.Drawing.Color.LightGray; 
            }
            else 
            {
                _imageCaptureControlMsg.Clear();
                _imageCaptureControlMsg.ImageCaptureEnabled = true;
                if (cbContinuousCapture.Checked)
                {
                    _imageCaptureControlMsg.NumberOfImagesToCapture = 0;
                }
                else
                {
                    _imageCaptureControlMsg.NumberOfImagesToCapture = (UInt32)nbxNumerOfImgesToCapture.Value;
                }
                VisionCmdProc.SendImageCaptureControlCmd(_imageCaptureControlMsg);
                _imageCaptureEnabledCmd = true;
                btEnableImageCapture.Text = "Disable";
                btEnableImageCapture.BackColor = System.Drawing.Color.Green; 
            }
        }

        private void SetEnableImageCaptureButtonDisplayStatus()
        {
            if(VSImageCaptureEnabled)
            {
                if( !VSImageCaptureComplete 
                    && !VSEndOfImages 
                    && VSImageCaptureError == ImageCaptureError_e.None)
                {
                    btEnableImageCapture.Text = "Disable";
                    btEnableImageCapture.BackColor = System.Drawing.Color.LightGreen; 
                }
                else if( VSImageCaptureComplete )
                {
                    btEnableImageCapture.Text = "Done";
                    btEnableImageCapture.BackColor = System.Drawing.Color.Aquamarine; 
                }
                else if( VSEndOfImages )
                {
                    btEnableImageCapture.Text = "EndOfImgs";
                    btEnableImageCapture.BackColor = System.Drawing.Color.Aqua; 
                }
                else if (VSImageCaptureError != ImageCaptureError_e.None)
                {
                    btEnableImageCapture.Text = "Error";
                    btEnableImageCapture.BackColor = System.Drawing.Color.Red; 
                }
            }
            else
            {
                btEnableImageCapture.Text = "Enable";
                btEnableImageCapture.BackColor = System.Drawing.Color.Gray; 
            }
        }


        public void ProcessImageCaptureStatusMsg(ImageCaptureStatusMsg msg)
        {
            //Display Status Info as needed.
            VSImageCaptureEnabled = msg.ImageCaptureEnabled;
            VSImageCaptureComplete = msg.ImageCaptureComplete;
            VSEndOfImages = msg.EndOfImages;
            VSImageCaptureSource = msg.ImageCaptureSource;
            VSImageCaptureError = msg.ErrorCode;

            SetEnableImageCaptureButtonDisplayStatus();
        }
    }
}
