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
    public partial class ImageCaptureSetupAndStatus : UserControl
    {

        public ImageCaptureSource_e ImageCaptureSource = ImageCaptureSource_e.NoChange;

        private bool _imageCaptureEnabledCmd = false;

        private string openCVWebCamDevice = "0";

        private string IPM_Directory = "RecordedImages";

        private string CImgDirectory = "ImageFiles";
        private string CImgFileExt = "jpg";

        private string VideoDirectory = "VideoFiles";
        private string VideoFilename = "video.avi";

        private UInt32 _imageWidth = 640;
        private UInt32 _imageHeight = 480;

        public VisionCmdProcess VisionCmdProc;


        public bool VSImageCaptureEnabled = false;

        public bool VSImageCaptureComplete = false;

        public bool VSEndOfImages = false;

        public ImageCaptureSource_e VSImageCaptureSource = ImageCaptureSource_e.NoChange;

        public ImageCaptureError_e VSImageCaptureError = ImageCaptureError_e.None;


        private ImageCaptureControlMsg _imageCaptureControlMsg;
        private ImageCaptureControlMsg _imageCaptureControlStatusMsg;

        private bool _ignoreSChange = false;




        public ImageCaptureSetupAndStatus()
        {
            InitializeComponent();
            _imageCaptureControlMsg = new ImageCaptureControlMsg();
            _imageCaptureControlStatusMsg = new ImageCaptureControlMsg();

            cbImageCaptureSource.Items.Add(ImageCaptureSource_e.NoChange);
            cbImageCaptureSource.Items.Add(ImageCaptureSource_e.OpenCVWebCam);
            cbImageCaptureSource.Items.Add(ImageCaptureSource_e.ImagePlusMetadataFiles);
            cbImageCaptureSource.Items.Add(ImageCaptureSource_e.CompressedImages);
            cbImageCaptureSource.Items.Add(ImageCaptureSource_e.Sensoray2253);
            cbImageCaptureSource.Items.Add(ImageCaptureSource_e.NVidiaCSIcam);
            cbImageCaptureSource.Items.Add(ImageCaptureSource_e.VideoFile);
            _ignoreSChange = true;  //Prevent new setup at startup.
            cbImageCaptureSource.SelectedIndex = 1;

            cbImageFormat.Items.Add(CPImageCaptureFormat_e.Unknown);
            cbImageFormat.Items.Add(CPImageCaptureFormat_e.Grey8);
            cbImageFormat.Items.Add(CPImageCaptureFormat_e.Grey16);
            cbImageFormat.Items.Add(CPImageCaptureFormat_e.RGB24);
            cbImageFormat.Items.Add(CPImageCaptureFormat_e.BGR24);
            cbImageFormat.Items.Add(CPImageCaptureFormat_e.MJPEG);
            cbImageFormat.Items.Add(CPImageCaptureFormat_e.YUV422);
            cbImageFormat.SelectedIndex = 6;


            btnICEnableDisplay.Text = "Disabled";
            btnICEnableDisplay.BackColor = System.Drawing.Color.LightGray;
        }


        private void setOpenCVWebcamCfg()
        {
            lblmageCapSourceP1.Text = "WebCam Device [0, 1...] (Empty-->Cfg Used)";
            tbImageCapSourceP1.Text = openCVWebCamDevice;
            lblmageCapSourceP1.Visible = true;
            tbImageCapSourceP1.Visible = true;

            lblmageCapSourceP2.Text = "Frame Rate";
            tblmageCapSourceP2.Text = "30";
            lblmageCapSourceP2.Visible = true;
            tblmageCapSourceP2.Visible = true;

            cbImageFormat.Visible = true;
            lblImageFormat.Visible = true;
            cbAutoFocusEnable.Visible = true;

            lbImageWidth.Visible = true;
            tbImageWidth.Visible = true;
            tbImageWidth.Text = _imageWidth.ToString();

            lbImageHeight.Visible = true;
            tbImageHeight.Visible = true;
            tbImageHeight.Text = _imageHeight.ToString();

            cbLoopImages.Visible = false;
            ImageCaptureSource = ImageCaptureSource_e.OpenCVWebCam;
        }

        private void setNVidiaCSIcamCfg()
        {
            lblmageCapSourceP1.Text = "WebCam Device [0, 1...] (Empty-->Cfg Used)";
            tbImageCapSourceP1.Text = "0";
            lblmageCapSourceP1.Visible = false;
            tbImageCapSourceP1.Visible = false;

            lblmageCapSourceP2.Text = "Frame Rate";
            tblmageCapSourceP2.Text = "30";
            lblmageCapSourceP2.Visible = true;
            tblmageCapSourceP2.Visible = true;

            cbImageFormat.Visible = false;
            lblImageFormat.Visible = false;
            cbAutoFocusEnable.Visible = false;

            lbImageWidth.Visible = true;
            tbImageWidth.Visible = true;
            tbImageWidth.Text = _imageWidth.ToString();

            lbImageHeight.Visible = true;
            tbImageHeight.Visible = true;
            tbImageHeight.Text = _imageHeight.ToString();

            cbLoopImages.Visible = false;
            ImageCaptureSource = ImageCaptureSource_e.NVidiaCSIcam;
        }

        private void setSensoray2253Cfg()
        {
            lblmageCapSourceP1.Text = "Device [0, 1] (Empty-->Cfg Used)";
            tbImageCapSourceP1.Text = openCVWebCamDevice;
            lblmageCapSourceP1.Visible = true;
            tbImageCapSourceP1.Visible = true;

            lblmageCapSourceP2.Visible = false;
            tblmageCapSourceP2.Visible = false;

            lbImageWidth.Visible = true;
            tbImageWidth.Visible = true;
            tbImageWidth.Text = _imageWidth.ToString();

            lbImageHeight.Visible = true;
            tbImageHeight.Visible = true;
            tbImageHeight.Text = _imageHeight.ToString();

            cbImageFormat.Visible = true;
            lblImageFormat.Visible = true;
            cbAutoFocusEnable.Visible = true;

            cbLoopImages.Visible = false;
            ImageCaptureSource = ImageCaptureSource_e.Sensoray2253;
        }

        private void setIPMCfg()
        {
            lblmageCapSourceP1.Text = "IPM Directory (Empty-->Cfg Used)";
            tbImageCapSourceP1.Text = IPM_Directory;
            lblmageCapSourceP1.Visible = true;
            tbImageCapSourceP1.Visible = true;

            lblmageCapSourceP2.Visible = false;
            tblmageCapSourceP2.Visible = false;

            lbImageWidth.Visible = false;
            tbImageWidth.Visible = false;

            lbImageHeight.Visible = false;
            tbImageHeight.Visible = false;

            cbImageFormat.Visible = false;
            lblImageFormat.Visible = false;
            cbAutoFocusEnable.Visible = false;

            cbLoopImages.Visible = true;
            ImageCaptureSource = ImageCaptureSource_e.ImagePlusMetadataFiles;
        }

        private void setCImgCfg()
        {
            lblmageCapSourceP1.Text = "Image Directory (Empty-->Cfg Used)";
            tbImageCapSourceP1.Text = CImgDirectory;
            lblmageCapSourceP1.Visible = true;
            tbImageCapSourceP1.Visible = true;

            lblmageCapSourceP2.Text = "File Ext";
            tblmageCapSourceP2.Text = CImgFileExt;
            lblmageCapSourceP2.Visible = true;
            tblmageCapSourceP2.Visible = true;

            lbImageWidth.Visible = false;
            tbImageWidth.Visible = false;

            lbImageHeight.Visible = false;
            tbImageHeight.Visible = false;

            cbImageFormat.Visible = false;
            lblImageFormat.Visible = false;
            cbAutoFocusEnable.Visible = false;

            cbLoopImages.Visible = true;
            ImageCaptureSource = ImageCaptureSource_e.CompressedImages;
        }

        private void setVideoFileCfg()
        {
            lblmageCapSourceP1.Text = "Video Directory (Empty-->Cfg Used)";
            tbImageCapSourceP1.Text = VideoDirectory;
            lblmageCapSourceP1.Visible = true;
            tbImageCapSourceP1.Visible = true;

            lblmageCapSourceP2.Text = "File Name";
            tblmageCapSourceP2.Text = VideoFilename;
            lblmageCapSourceP2.Visible = true;
            tblmageCapSourceP2.Visible = true;

            lbImageWidth.Visible = false;
            tbImageWidth.Visible = false;

            lbImageHeight.Visible = false;
            tbImageHeight.Visible = false;

            cbImageFormat.Visible = false;
            lblImageFormat.Visible = false;
            cbAutoFocusEnable.Visible = false;

            cbLoopImages.Visible = true;
            ImageCaptureSource = ImageCaptureSource_e.VideoFile;
        }

        private void cbImageCaptureSource_SelectedIndexChanged(object sender, EventArgs e)
        {
            switch ((ImageCaptureSource_e)cbImageCaptureSource.SelectedItem)
            {
                case ImageCaptureSource_e.NoChange:
                    _ignoreSChange = true;
                    break;
                case ImageCaptureSource_e.OpenCVWebCam:
                    setOpenCVWebcamCfg();
                    break;
                case ImageCaptureSource_e.ImagePlusMetadataFiles:
                    setIPMCfg();
                    break;
                case ImageCaptureSource_e.CompressedImages:
                    setCImgCfg();
                    break;
                case ImageCaptureSource_e.Sensoray2253:
                    setSensoray2253Cfg();
                    break;
                case ImageCaptureSource_e.NVidiaCSIcam:
                    setNVidiaCSIcamCfg();
                    break;
                case ImageCaptureSource_e.VideoFile:
                    setVideoFileCfg();
                    break;
            }
            if (!_ignoreSChange)
            {
                //Action only if ... 
            }
            _ignoreSChange = false;
        }

        public void SetupCaptureSource()
        {
            UInt32 frameRate = 30;
            //The Capture source can only be changed with Image Capture 
            //Disabled.
            _imageCaptureControlMsg.Clear();
            _imageCaptureControlMsg.ImageCaptureEnabled = false;

            switch ((ImageCaptureSource_e)cbImageCaptureSource.SelectedItem)
            {
                case ImageCaptureSource_e.NoChange:
                    return;
                    break;
                case ImageCaptureSource_e.OpenCVWebCam:
                    _imageCaptureControlMsg.ImageCaptureSource = ImageCaptureSource_e.OpenCVWebCam;
                    openCVWebCamDevice = tbImageCapSourceP1.Text;
                    _imageCaptureControlMsg.ImageCaptureSourceConfigPri = openCVWebCamDevice;
                    _imageCaptureControlMsg.ImageCaptureSourceConfigSec = "";

                    frameRate = 30;
                    UInt32.TryParse(tblmageCapSourceP2.Text, out frameRate);
                    _imageCaptureControlMsg.DesiredFramesPerSecond = frameRate;

                    UInt32.TryParse(tbImageWidth.Text, out _imageWidth);
                    _imageCaptureControlMsg.DesiredImageWidth = _imageWidth;
                    UInt32.TryParse(tbImageHeight.Text, out _imageHeight);
                    _imageCaptureControlMsg.DesiredImageHeight = _imageHeight;
                    _imageCaptureControlMsg.ImageCaptureFormat = (CPImageCaptureFormat_e)cbImageFormat.SelectedItem;
                    _imageCaptureControlMsg.AutoFocusEnable = cbAutoFocusEnable.Checked;
                    break;
                case ImageCaptureSource_e.NVidiaCSIcam:
                    _imageCaptureControlMsg.ImageCaptureSource = ImageCaptureSource_e.NVidiaCSIcam;
                    _imageCaptureControlMsg.ImageCaptureSourceConfigPri = "0";
                    _imageCaptureControlMsg.ImageCaptureSourceConfigSec = "";
                    frameRate = 30;
                    UInt32.TryParse(tblmageCapSourceP2.Text, out frameRate);
                    _imageCaptureControlMsg.DesiredFramesPerSecond = frameRate;
                    UInt32.TryParse(tbImageWidth.Text, out _imageWidth);
                    _imageCaptureControlMsg.DesiredImageWidth = _imageWidth;
                    UInt32.TryParse(tbImageHeight.Text, out _imageHeight);
                    _imageCaptureControlMsg.DesiredImageHeight = _imageHeight;
                    _imageCaptureControlMsg.ImageCaptureFormat = (CPImageCaptureFormat_e)cbImageFormat.SelectedItem;
                    _imageCaptureControlMsg.AutoFocusEnable = cbAutoFocusEnable.Checked;
                    break;

                case ImageCaptureSource_e.ImagePlusMetadataFiles:
                    _imageCaptureControlMsg.ImageCaptureSource = ImageCaptureSource_e.ImagePlusMetadataFiles;
                    IPM_Directory = tbImageCapSourceP1.Text;
                    _imageCaptureControlMsg.ImageCaptureSourceConfigPri = IPM_Directory;
                    _imageCaptureControlMsg.ImageCaptureSourceConfigSec = "";
                    _imageCaptureControlMsg.ImageSourceLoopAround = cbLoopImages.Checked;
                    break;
                case ImageCaptureSource_e.CompressedImages:
                    _imageCaptureControlMsg.ImageCaptureSource = ImageCaptureSource_e.CompressedImages;
                    CImgDirectory = tbImageCapSourceP1.Text;
                    _imageCaptureControlMsg.ImageCaptureSourceConfigPri = CImgDirectory;
                    CImgFileExt = tblmageCapSourceP2.Text;
                    _imageCaptureControlMsg.ImageCaptureSourceConfigSec = CImgFileExt;
                    _imageCaptureControlMsg.ImageSourceLoopAround = cbLoopImages.Checked;
                    break;
                case ImageCaptureSource_e.Sensoray2253:
                    _imageCaptureControlMsg.ImageCaptureSource = ImageCaptureSource_e.Sensoray2253;
                    openCVWebCamDevice = tbImageCapSourceP1.Text;
                    _imageCaptureControlMsg.ImageCaptureSourceConfigPri = openCVWebCamDevice;
                    _imageCaptureControlMsg.ImageCaptureSourceConfigSec = "";
                    _imageCaptureControlMsg.ImageCaptureFormat = (CPImageCaptureFormat_e)cbImageFormat.SelectedItem;
                    _imageCaptureControlMsg.AutoFocusEnable = cbAutoFocusEnable.Checked;

                    UInt32.TryParse(tbImageWidth.Text, out _imageWidth);
                    _imageCaptureControlMsg.DesiredImageWidth = _imageWidth;
                    UInt32.TryParse(tbImageHeight.Text, out _imageHeight);
                    _imageCaptureControlMsg.DesiredImageHeight = _imageHeight;
                    break;
                case ImageCaptureSource_e.VideoFile:
                    _imageCaptureControlMsg.ImageCaptureSource = ImageCaptureSource_e.VideoFile;
                    VideoDirectory = tbImageCapSourceP1.Text;
                    _imageCaptureControlMsg.ImageCaptureSourceConfigPri = VideoDirectory;
                    VideoFilename = tblmageCapSourceP2.Text;
                    _imageCaptureControlMsg.ImageCaptureSourceConfigSec = VideoFilename;
                    _imageCaptureControlMsg.ImageSourceLoopAround = cbLoopImages.Checked;
                    break;

            }

            VisionCmdProc.SendImageCaptureControlCmd(_imageCaptureControlMsg);
        }

        public ImageCaptureControlMsg GetImageCaptureControlStatus()
        {
            string respMsg;
            return VisionCmdProc.GetImageCaptureControlSettings(out respMsg);
        }

        private void btSetImageSource_Click(object sender, EventArgs e)
        {
            SetupCaptureSource();
            //Get the status.
            //_imageCaptureControlStatusMsg = GetImageCaptureControlStatus();
            //DisplayImageCaptureControlStatusMsg(_imageCaptureControlStatusMsg);
        }

        public void ProcessImageCaptureStatusMsg(ImageCaptureStatusMsg msg)
        {
            //Display Status Info as needed.
            VSImageCaptureEnabled = msg.ImageCaptureEnabled;
            VSImageCaptureComplete = msg.ImageCaptureComplete;
            VSEndOfImages = msg.EndOfImages;
            VSImageCaptureSource = msg.ImageCaptureSource;
            VSImageCaptureError = msg.ErrorCode;

            tbActCaptureSource.Text = VSImageCaptureSource.ToString();
            if (VSImageCaptureError != ImageCaptureError_e.None
                || VSImageCaptureSource == ImageCaptureSource_e.NoChange)
            {
                tbActCaptureSource.BackColor = System.Drawing.Color.LightSalmon;
            }
            else
            {
                tbActCaptureSource.BackColor = System.Drawing.Color.LightGreen;
            }

            tbNumerImagesCaptured.Text = msg.CurrentNumberOfImagesCaptured.ToString();
            tbTotalNoImagesCaptured.Text = msg.TotalNumberOfImagesCaptured.ToString();
            tbAveFrameRate.Text = msg.AverageFramesPerSecond.ToString("0.0");

            SetEnableImageCaptureDisplayStatus();
        }

        public void DisplayImageCaptureControlStatusMsg(ImageCaptureControlMsg msg)
        {
            if (msg != null)
            {
                cbImageCaptureSource.SelectedItem = msg.ImageCaptureSource;
                //cbImageFormat.SelectedItem = msg.ImageCaptureFormat;
                tbImageCapSourceP1.Text = msg.ImageCaptureSourceConfigPri;
                tblmageCapSourceP2.Text = msg.ImageCaptureSourceConfigSec;
                cbAutoFocusEnable.Checked = msg.AutoFocusEnable;
                cbLoopImages.Checked = msg.ImageSourceLoopAround;
                tbImageHeight.Text = msg.DesiredImageHeight.ToString();
                tbImageWidth.Text = msg.DesiredImageWidth.ToString();
                tbAveFrameRate.Text = msg.DesiredFramesPerSecond.ToString();
            }
        }

 
        private void SetEnableImageCaptureDisplayStatus()
        {
            if(VSImageCaptureEnabled)
            {
                if( !VSImageCaptureComplete 
                    && !VSEndOfImages 
                    && VSImageCaptureError == ImageCaptureError_e.None)
                {
                    btnICEnableDisplay.Text = "Enabled";
                    btnICEnableDisplay.BackColor = System.Drawing.Color.LightGreen; 
                }
                else if( VSImageCaptureComplete )
                {
                    btnICEnableDisplay.Text = "Capture Done";
                    btnICEnableDisplay.BackColor = System.Drawing.Color.Aquamarine; 
                }
                else if( VSEndOfImages )
                {
                    btnICEnableDisplay.Text = "End Of Images";
                    btnICEnableDisplay.BackColor = System.Drawing.Color.Aqua; 
                }
                else if (VSImageCaptureError != ImageCaptureError_e.None)
                {
                    btnICEnableDisplay.Text = "Error";
                    btnICEnableDisplay.BackColor = System.Drawing.Color.Red; 
                }
            }
            else
            {
                btnICEnableDisplay.Text = "Disabled";
                btnICEnableDisplay.BackColor = System.Drawing.Color.Gray; 
            }
        }

        private void btnGetImageCaptureInfo_Click(object sender, EventArgs e)
        {
            _imageCaptureControlStatusMsg = GetImageCaptureControlStatus();
            DisplayImageCaptureControlStatusMsg(_imageCaptureControlStatusMsg);
        }

 
   }

}
