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
    public partial class HeadTrackingControlWidget : UserControl
    {

        public VisionCmdProcess VisionCmdProc;

        bool ignoreChange = false;

        HeadTrackingControlPBMsg HeadTrackingControlMsg;

        public HeadTrackingControlWidget()
        {
            InitializeComponent();

            HeadTrackingControlMsg = new HeadTrackingControlPBMsg();

            cBoxDisplayType.Items.Add("None");
            cBoxDisplayType.Items.Add("Glyph Contours");
            cBoxDisplayType.Items.Add("Orientation");
            ignoreChange = true;
            cBoxDisplayType.SelectedIndex = 2;

            cmbBxGlyphModelNo.Items.Add(1);
            cmbBxGlyphModelNo.Items.Add(2);
            cmbBxGlyphModelNo.Items.Add(3);
            ignoreChange = true;
            cmbBxGlyphModelNo.SelectedIndex = 0;

            ignoreChange = false;
        }

        public void SetDefaultParameters()
        {
            tbCannyLow.Text = "50";
            ignoreChange = true;
            hSBCannyLow.Value = 50;
            HeadTrackingControlMsg.Canny_low = 50;

            tbCannyHigh.Text = "150";
            ignoreChange = true;
            hSBCannyHigh.Value = 150;
            HeadTrackingControlMsg.Canny_high = 150;

            tbGlyphAreaMin.Text = "1000";
            ignoreChange = true;
            hSBGlyphAreaMin.Value = 1000;
            HeadTrackingControlMsg.GlyphAreaPixels_min = 1000;

            tbGlyphAreaMax.Text = "8000";
            ignoreChange = true;
            hSBGlyphAreaMax.Value = 8000;
            HeadTrackingControlMsg.GlyphAreaPixels_max = 8000;

            tbNoIterations.Text = "10";
            ignoreChange = true;
            hSBNoIterations.Value = 10;
            HeadTrackingControlMsg.NumberOfIterations = 10;

            tbReprojectionError.Text = "5.0";
            ignoreChange = true;
            hSBReprojectionError.Value = 5;
            HeadTrackingControlMsg.ReprojectionErrorDistance = 5.0;

            tbhSBConfidencePercent.Text = "95.0";
            ignoreChange = true;
            hSBConfidencePercent.Value = 95;
            HeadTrackingControlMsg.ConfidencePercent = 95.0;

            HeadTrackingControlMsg.HeadTrackingDisplayType = (UInt32)cBoxDisplayType.SelectedIndex;
        }


        public void readAllParameters()
        {
            double dval = 0;
            int ival = 0;

            if (int.TryParse(tbCannyLow.Text, out ival))
            {
                HeadTrackingControlMsg.Canny_low = ival;
                ignoreChange = true;
                hSBCannyLow.Value = ival;
            }
            if (int.TryParse(tbCannyHigh.Text, out ival))
            {
                HeadTrackingControlMsg.Canny_high = ival;
                ignoreChange = true;
                hSBCannyHigh.Value = ival;
            }
            if (int.TryParse(tbGlyphAreaMin.Text, out ival))
            {
                HeadTrackingControlMsg.GlyphAreaPixels_min = ival;
                ignoreChange = true;
                hSBGlyphAreaMin.Value = ival;
            }
            if (int.TryParse(tbGlyphAreaMax.Text, out ival))
            {
                HeadTrackingControlMsg.GlyphAreaPixels_max = ival;
                ignoreChange = true;
                hSBGlyphAreaMax.Value = ival;
            }
            if (int.TryParse(tbNoIterations.Text, out ival))
            {
                HeadTrackingControlMsg.NumberOfIterations = ival;
                ignoreChange = true;
                hSBNoIterations.Value = ival;
            }
            if (double.TryParse(tbReprojectionError.Text, out dval))
            {
                dval = dval < 1.0 ? 1.0 : dval > 100.0 ? 100.0 : dval;
                HeadTrackingControlMsg.ReprojectionErrorDistance = dval;
                ignoreChange = true;
                hSBReprojectionError.Value = (int)(1.0 * dval);
            }

            if (double.TryParse(tbhSBConfidencePercent.Text, out dval))
            {
                dval = dval < 1.0 ? 1.0 : dval > 100.0 ? 100.0 : dval;
                HeadTrackingControlMsg.ConfidencePercent = dval;
                ignoreChange = true;
                hSBConfidencePercent.Value = (int)(dval);
            }

            HeadTrackingControlMsg.HeadTrackingDisplayType = (UInt32)cBoxDisplayType.SelectedIndex;
            HeadTrackingControlMsg.GlyphModelIndex = (UInt32)cmbBxGlyphModelNo.SelectedIndex;
        }

        private void btnSend_Click(object sender, EventArgs e)
        {
            readAllParameters();
            VisionCmdProc.SendHeadTrackingControlMsg(HeadTrackingControlMsg);
        }

        private void hSBCannyLow_Scroll(object sender, ScrollEventArgs e)
        {
            if (!ignoreChange)
            {
                HeadTrackingControlMsg.Canny_low = hSBCannyLow.Value;
                tbCannyLow.Text = HeadTrackingControlMsg.Canny_low.ToString();
                VisionCmdProc.SendHeadTrackingControlMsg(HeadTrackingControlMsg);
            }
            ignoreChange = false;
        }

        private void hSBCannyHigh_Scroll(object sender, ScrollEventArgs e)
        {
            if (!ignoreChange)
            {
                HeadTrackingControlMsg.Canny_high = hSBCannyHigh.Value;
                tbCannyHigh.Text = HeadTrackingControlMsg.Canny_high.ToString();
                VisionCmdProc.SendHeadTrackingControlMsg(HeadTrackingControlMsg);
            }
            ignoreChange = false;
        }

        private void hSBGlyphAreaMin_Scroll(object sender, ScrollEventArgs e)
        {
            if (!ignoreChange)
            {
                HeadTrackingControlMsg.GlyphAreaPixels_min = hSBGlyphAreaMin.Value;
                tbGlyphAreaMin.Text = HeadTrackingControlMsg.GlyphAreaPixels_min.ToString();
                VisionCmdProc.SendHeadTrackingControlMsg(HeadTrackingControlMsg);
            }
            ignoreChange = false;
        }

        private void hSBGlyphAreaMax_Scroll(object sender, ScrollEventArgs e)
        {
            if (!ignoreChange)
            {
                HeadTrackingControlMsg.GlyphAreaPixels_max = hSBGlyphAreaMax.Value;
                tbGlyphAreaMax.Text = HeadTrackingControlMsg.GlyphAreaPixels_max.ToString();
                VisionCmdProc.SendHeadTrackingControlMsg(HeadTrackingControlMsg);
            }
            ignoreChange = false;
        }

        private void hSBNoIterations_Scroll(object sender, ScrollEventArgs e)
        {
            if (!ignoreChange)
            {
                HeadTrackingControlMsg.NumberOfIterations = hSBNoIterations.Value;
                tbNoIterations.Text = HeadTrackingControlMsg.NumberOfIterations.ToString();
                VisionCmdProc.SendHeadTrackingControlMsg(HeadTrackingControlMsg);
            }
            ignoreChange = false;
        }

        private void hSBReprojectionError_Scroll(object sender, ScrollEventArgs e)
        {
            if (!ignoreChange)
            {
                HeadTrackingControlMsg.ReprojectionErrorDistance = 1.0 * (double)hSBReprojectionError.Value;
                tbReprojectionError.Text = HeadTrackingControlMsg.ReprojectionErrorDistance.ToString("0.00");
                VisionCmdProc.SendHeadTrackingControlMsg(HeadTrackingControlMsg);
            }
            ignoreChange = false;
        }

        private void hSBConfidencePercent_Scroll(object sender, ScrollEventArgs e)
        {
            if (!ignoreChange)
            {
                HeadTrackingControlMsg.ConfidencePercent = (double)hSBConfidencePercent.Value;
                tbhSBConfidencePercent.Text = HeadTrackingControlMsg.ConfidencePercent.ToString("0.0");
                VisionCmdProc.SendHeadTrackingControlMsg(HeadTrackingControlMsg);
            }
            ignoreChange = false;
        }

        private void cBoxDisplayType_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (!ignoreChange)
            {
                HeadTrackingControlMsg.HeadTrackingDisplayType = (UInt32)cBoxDisplayType.SelectedIndex;
                VisionCmdProc.SendHeadTrackingControlMsg(HeadTrackingControlMsg);
            }
            ignoreChange = false;
        }

        private void cmbBxGlyphModelNo_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (!ignoreChange)
            {
                HeadTrackingControlMsg.GlyphModelIndex = (UInt32)cmbBxGlyphModelNo.SelectedIndex;
                VisionCmdProc.SendHeadTrackingControlMsg(HeadTrackingControlMsg);
            }
            ignoreChange = false;
        }
    }
}
