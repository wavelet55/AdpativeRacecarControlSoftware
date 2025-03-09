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

    //Reference:  https://en.wikipedia.org/wiki/HSL_and_HSV
    public enum ColorFormat_e
    {
        RGB,
        HSV,        //Hue, Saturation, Value:  V = Max(R,G,B)
        HSL,        //Hue, Saturation, Lumanance:  L = 0.5 * [Max(R,G,B) + Min(R,G,B)]
        YCrCb
    }

    //Notes: on Color Min/Max Settings.   
    //The Windows Color Selection dialog allows setting colors in RGB or HSL format
    //but only returns colors in RGB format.  
    //Hue is defined to be in the range of [0, 360.0) degrees.  Red colors span 
    //the 360 degree values... in other words 0/360.0 are about the center of the 
    //red color spectrum.  To encode min to max color values, for Hue, if the min
    //hue value is > max Hue, then the Hue range is assumed to be Hue_min increasing
    //to 360 and wrapping around to 0 up to the Hue_max value.  If Hue_min is < Hue_max
    //then the color range is those values bewteen Hue_min and Hue_max (inclusive).

    public partial class BlobDetectorParameters : UserControl
    {
        public VisionCmdProcess VisionCmdProc;

        private PixelColorValue_t _minBlobColor;
        private PixelColorValue_t _maxBlobColor;

        //System.Drawing.Color _minBlobColor = System.Drawing.Color.LightPink;
        //System.Drawing.Color _maxBlobColor = System.Drawing.Color.Red;

        public BlobDetectorParameters()
        {
            InitializeComponent();
            cbGaussianFilterPixelRange.Items.Add(0);
            for (int i = 3; i < 26; i += 2)
            {
                cbGaussianFilterPixelRange.Items.Add(i);
            }
            cbGaussianFilterPixelRange.SelectedIndex = 0;

            cbImageOutDispOption.Items.Add("None");
            cbImageOutDispOption.Items.Add("B&W Blobs");
            cbImageOutDispOption.Items.Add("Circle Tgts");
            cbImageOutDispOption.Items.Add("Circle All");
            cbImageOutDispOption.SelectedIndex = 0;

            _minBlobColor = new PixelColorValue_t();
            _maxBlobColor = new PixelColorValue_t();
        }

        private void btnSelBlobColor_Click(object sender, EventArgs e)
        {
            ColorSelectionForm colorSelectForm = new ColorSelectionForm();
            colorSelectForm.MinColor = _minBlobColor;
            colorSelectForm.MaxColor = _maxBlobColor;
            colorSelectForm.ShowDialog();
            if (colorSelectForm.ColorsUpdated)
            {
                _minBlobColor.CopyFrom(colorSelectForm.MinColor);
                _maxBlobColor.CopyFrom(colorSelectForm.MaxColor);
            }
            colorSelectForm.Dispose();
        }


        //A zero in the Most significant byte indicates the 
        //color values are in RGB Format.  In the future.. different
        //formats could be send to Videre.
        //The color values are currently sent to Videre in RGB format
        //Videre will convert to the correct format for blob color 
        //comparisons.
        public Int32 ColorToInt(System.Drawing.Color color)
        {
            Int32 cVal = 0;
            cVal = color.R;
            cVal = (cVal << 8) + color.G;
            cVal = (cVal << 8) + color.B;
            return cVal;
        }


        public void GetBlobDetectorParameters(FeatureMatchProcCtrlPBMsg blobParamsMsg)
        {
            int intVal;
            double dVal;
            blobParamsMsg.ParamI_1 = (int)cbGaussianFilterPixelRange.SelectedItem;

            double.TryParse(tbBlobMinArea.Text, out dVal);
            blobParamsMsg.ParamF_10 = dVal;
            double.TryParse(tbBlobMaxArea.Text, out dVal);
            blobParamsMsg.ParamF_11 = dVal;

            double.TryParse(tbMinDistBetweenBlobs.Text, out dVal);
            blobParamsMsg.ParamF_12 = dVal;

            blobParamsMsg.ParamI_2 = (int)_minBlobColor.ToUInt();
            blobParamsMsg.ParamI_3 = (int)_maxBlobColor.ToUInt();

            if (cbBlobCircularityEnabled.Checked)
            {
                double.TryParse(tbBlobMinCircularity.Text, out dVal);
                blobParamsMsg.ParamF_14 = dVal;
                double.TryParse(tbBlobMaxCircularity.Text, out dVal);
                blobParamsMsg.ParamF_15 = dVal;
            }
            else
            {
                blobParamsMsg.ParamF_14 = 0;
                blobParamsMsg.ParamF_15 = 0;
            }
            if (cbBlobConvexityEnabled.Checked)
            {
                double.TryParse(tbBlobMinConvexity.Text, out dVal);
                blobParamsMsg.ParamF_16 = dVal;
                double.TryParse(tbBlobMaxConvexity.Text, out dVal);
                blobParamsMsg.ParamF_17 = dVal;
            }
            else
            {
                blobParamsMsg.ParamF_16 = 0;
                blobParamsMsg.ParamF_17 = 0;
            }
            if (cbBlobInertialRatioEnabled.Checked)
            {
                double.TryParse(tbBlobMinInertialRatio.Text, out dVal);
                blobParamsMsg.ParamF_18 = dVal;
                double.TryParse(tbBlobMaxInertialRatio.Text, out dVal);
                blobParamsMsg.ParamF_19 = dVal;
            }
            else
            {
                blobParamsMsg.ParamF_18 = 0;
                blobParamsMsg.ParamF_19 = 0;
            }

            blobParamsMsg.FMImagePostProcessMethod = (FMImagePostProcessMethod_e)cbImageOutDispOption.SelectedIndex;
        }

        private void btnSendParameters_Click(object sender, EventArgs e)
        {
            FeatureMatchProcCtrlPBMsg blobParamsMsg = new FeatureMatchProcCtrlPBMsg();
            GetBlobDetectorParameters(blobParamsMsg);
            VisionCmdProc.SendFeatureMatchProcCtrlMsg(blobParamsMsg);
        }
    }
}
