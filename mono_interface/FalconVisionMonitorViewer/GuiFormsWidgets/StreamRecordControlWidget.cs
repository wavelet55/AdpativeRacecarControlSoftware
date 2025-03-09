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
    public partial class StreamRecordControlWidget : UserControl
    {
        public VisionCmdProcess VisionCmdProc;

        public StreamRecordControlWidget()
        {
            InitializeComponent();
        }

        private void btnSendMsg_Click(object sender, EventArgs e)
        {
            if (VisionCmdProc != null)
            {
                double fps = 5.0;
                UInt32 imgQlt = 50;
                double scaleFactor = 1.0;
                StreamControlPBMsg srcMsg = new StreamControlPBMsg();
                srcMsg.StreamImagesEnabled = cbStreamEnable.Checked;
                double.TryParse(tbStreamFPS.Text, out fps);
                fps = fps < 0.1 ? 0.1 : fps > 100.0 ? 100.0 : fps;
                tbStreamFPS.Text = fps.ToString("0.00");
                srcMsg.StreamImageFrameRate = fps;

                UInt32.TryParse(tbCompressedImgQuality.Text, out imgQlt);
                imgQlt = imgQlt < 10 ? 10 : imgQlt > 100 ? 100 : imgQlt;
                tbCompressedImgQuality.Text = imgQlt.ToString();
                srcMsg.ImageCompressionQuality = imgQlt;

                
                double.TryParse(tbImageScaleDownFactor.Text, out scaleFactor);
                scaleFactor = scaleFactor < 1.0 ? 1.0 : scaleFactor > 25.0 ? 25.0 : scaleFactor;
                tbImageScaleDownFactor.Text = scaleFactor.ToString("0.00");
                srcMsg.StreamImageScaleDownFactor = scaleFactor;

                VisionCmdProc.SendStreamRecordControlCmd(srcMsg);

                ImageLoggingControlMsg imgLogMsg = new ImageLoggingControlMsg();

                imgLogMsg.EnableLogging = cbRecordEnable.Checked;
                if (cbCompressRecordImageEnable.Checked)
                    imgLogMsg.LoggingType = ImageLoggingControlMsg.VisionLoggingType_e.LogCompressedImages;
                else
                    imgLogMsg.LoggingType = ImageLoggingControlMsg.VisionLoggingType_e.LogRawImages;

                VisionCmdProc.SendImageLoggingControlCmd(imgLogMsg);
            }
        }

        private void btnRetrieveMsg_Click(object sender, EventArgs e)
        {
            string rspMsg;
            StreamControlPBMsg srcMsg = VisionCmdProc.GetStreamRecordControlSettings(out rspMsg);
            if (srcMsg != null)
            {
                cbStreamEnable.Checked = srcMsg.StreamImagesEnabled;
                tbStreamFPS.Text = srcMsg.StreamImageFrameRate.ToString("0.00");
                tbCompressedImgQuality.Text = srcMsg.ImageCompressionQuality.ToString();
                tbImageScaleDownFactor.Text = srcMsg.StreamImageScaleDownFactor.ToString("0.00");

                ImageLoggingControlMsg imgLogMsg = VisionCmdProc.GetImageLoggingControlSettings(out rspMsg);
                cbRecordEnable.Checked = imgLogMsg.EnableLogging;
                cbCompressRecordImageEnable.Checked = imgLogMsg.LoggingType == ImageLoggingControlMsg.VisionLoggingType_e.LogCompressedImages;
            }
        }
    }
}
