using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    public partial class ColorSelectionForm : Form
    {
        private bool _colorsUpdated = false;
        public bool ColorsUpdated
        {
            get { return _colorsUpdated; }
        }

        //It is assumed that min and max Colors have the same format.
        private PixelColorValue_t _minColor;
        public PixelColorValue_t MinColor
        {
            get { return _minColor; }
            set
            {
                _minColor.CopyFrom(value);
                setColorFormat(_minColor.colorFormat);
            }
        }

        private PixelColorValue_t _maxColor;
        public PixelColorValue_t MaxColor
        {
            get { return _maxColor; }
            set
            {
                _maxColor.CopyFrom(value);
                setColorFormat(_maxColor.colorFormat);
            }
        }

        private bool _ignoreColorFmtChange = false;

        public ColorSelectionForm()
        {
            InitializeComponent();
            _minColor = new PixelColorValue_t();
            _maxColor = new PixelColorValue_t();

            cbColorFormat.Items.Add(ImageColorFormat_e.RGB.ToString());
            cbColorFormat.Items.Add(ImageColorFormat_e.HSV.ToString());
            cbColorFormat.Items.Add(ImageColorFormat_e.HSL.ToString());
            cbColorFormat.Items.Add(ImageColorFormat_e.HSI.ToString());
            cbColorFormat.Items.Add(ImageColorFormat_e.YCrCb.ToString());

            //This is the default format.
            setColorFormat(ImageColorFormat_e.RGB);
        }

        private void setColorFormat(ImageColorFormat_e colorFormat)
        {
            _ignoreColorFmtChange = true;
            cbColorFormat.SelectedIndex = (int)colorFormat;
            _ignoreColorFmtChange = false;

            if (_minColor.colorFormat != colorFormat)
                _minColor = _minColor.ChangeFormat(colorFormat);

            if (_maxColor.colorFormat != colorFormat)
                _maxColor = _maxColor.ChangeFormat(colorFormat);

            switch (colorFormat)
            {
                case ImageColorFormat_e.RGB:
                    lblColorName_c0.Text = "Red";
                    lblColorName_c1.Text = "Green";
                    lblColorName_c2.Text = "Blue";
                    break;

                case ImageColorFormat_e.HSV:
                    lblColorName_c0.Text = "Hue (Deg)";
                    lblColorName_c1.Text = "Sat (%)";
                    lblColorName_c2.Text = "Val (%)";
                    break;

                case ImageColorFormat_e.HSL:
                    lblColorName_c0.Text = "Hue (Deg)";
                    lblColorName_c1.Text = "Sat (%)";
                    lblColorName_c2.Text = "Lum (%)";
                    break;

                case ImageColorFormat_e.HSI:
                    lblColorName_c0.Text = "Hue (Deg)";
                    lblColorName_c1.Text = "Sat (%)";
                    lblColorName_c2.Text = "Int (%)";
                    break;

                case ImageColorFormat_e.YCrCb:
                    lblColorName_c0.Text = "Y";
                    lblColorName_c1.Text = "Cr";
                    lblColorName_c2.Text = "Cb";
                    break;
            }
            displayColorVals();
        }

        private void displayColorVals()
        {
            //It is assumed that both Min and Max color have the same format.
            switch (_minColor.colorFormat)
            {
                case ImageColorFormat_e.RGB:
                    tbMinColorVal_c0.Text = _minColor.Red.ToString();
                    tbMinColorVal_c1.Text = _minColor.Green.ToString();
                    tbMinColorVal_c2.Text = _minColor.Blue.ToString();
                    tbMaxColorVal_c0.Text = _maxColor.Red.ToString();
                    tbMaxColorVal_c1.Text = _maxColor.Green.ToString();
                    tbMaxColorVal_c2.Text = _maxColor.Blue.ToString();
                    break;

                case ImageColorFormat_e.HSV:
                case ImageColorFormat_e.HSL:
                case ImageColorFormat_e.HSI:
                    tbMinColorVal_c0.Text = _minColor.HueDegrees.ToString("0.0");
                    tbMinColorVal_c1.Text = _minColor.SaturationPercent.ToString("0.0");
                    tbMinColorVal_c2.Text = _minColor.VLIPercent.ToString("0.0");
                    tbMaxColorVal_c0.Text = _maxColor.HueDegrees.ToString("0.0");
                    tbMaxColorVal_c1.Text = _maxColor.SaturationPercent.ToString("0.0");
                    tbMaxColorVal_c2.Text = _maxColor.VLIPercent.ToString("0.0");
                    break;

                case ImageColorFormat_e.YCrCb:
                    tbMinColorVal_c0.Text = _minColor.Y.ToString();
                    tbMinColorVal_c1.Text = _minColor.Cr.ToString();
                    tbMinColorVal_c2.Text = _minColor.Cb.ToString();
                    tbMaxColorVal_c0.Text = _maxColor.Y.ToString();
                    tbMaxColorVal_c1.Text = _maxColor.Cr.ToString();
                    tbMaxColorVal_c2.Text = _maxColor.Cb.ToString();
                    break;
            }
        }


        private void readColorVals()
        {
            byte byteVal = 0;
            double dval = 0;

                //It is assumed that both Min and Max color have the same format.
                switch (_minColor.colorFormat)
                {
                    case ImageColorFormat_e.RGB:
                        byte.TryParse(tbMinColorVal_c0.Text, out byteVal);
                        _minColor.Red = byteVal;
                        byte.TryParse(tbMinColorVal_c1.Text, out byteVal);
                        _minColor.Green = byteVal;
                        byte.TryParse(tbMinColorVal_c2.Text, out byteVal);
                        _minColor.Blue = byteVal;
                        byte.TryParse(tbMaxColorVal_c0.Text, out byteVal);
                        _maxColor.Red = byteVal;
                        byte.TryParse(tbMaxColorVal_c1.Text, out byteVal);
                        _maxColor.Green = byteVal;
                        byte.TryParse(tbMaxColorVal_c2.Text, out byteVal);
                        _maxColor.Blue = byteVal;
                       break;

                    case ImageColorFormat_e.HSV:
                    case ImageColorFormat_e.HSL:
                    case ImageColorFormat_e.HSI:
                        double.TryParse(tbMinColorVal_c0.Text, out dval);
                        _minColor.HueDegrees = dval;
                        double.TryParse(tbMinColorVal_c1.Text, out dval);
                        _minColor.SaturationPercent = dval;
                        double.TryParse(tbMinColorVal_c2.Text, out dval);
                        _minColor.VLIPercent = dval;
                        double.TryParse(tbMaxColorVal_c0.Text, out dval);
                        _maxColor.HueDegrees = dval;
                        double.TryParse(tbMaxColorVal_c1.Text, out dval);
                        _maxColor.SaturationPercent = dval;
                        double.TryParse(tbMaxColorVal_c2.Text, out dval);
                        _maxColor.VLIPercent = dval;
                        break;

                    case ImageColorFormat_e.YCrCb:
                        byte.TryParse(tbMinColorVal_c0.Text, out byteVal);
                        _minColor.Y = byteVal;
                        byte.TryParse(tbMinColorVal_c1.Text, out byteVal);
                        _minColor.Cr = byteVal;
                        byte.TryParse(tbMinColorVal_c2.Text, out byteVal);
                        _minColor.Cb = byteVal;
                        byte.TryParse(tbMaxColorVal_c0.Text, out byteVal);
                        _maxColor.Y = byteVal;
                        byte.TryParse(tbMaxColorVal_c1.Text, out byteVal);
                        _maxColor.Cr = byteVal;
                        byte.TryParse(tbMaxColorVal_c2.Text, out byteVal);
                        _maxColor.Cb = byteVal;
                       break;
                        break;
                }
        }


        private void btnMaxColorDialog_Click(object sender, EventArgs e)
        {
            if(colorDialogBox.ShowDialog() == DialogResult.OK)  
            {
                PixelColorValue_t pcv = new PixelColorValue_t();
                pcv.colorFormat = ImageColorFormat_e.RGB;
                pcv.Red = colorDialogBox.Color.R;
                pcv.Green = colorDialogBox.Color.G;
                pcv.Blue = colorDialogBox.Color.B;
                _maxColor = pcv.ChangeFormat(_maxColor.colorFormat);
                displayColorVals();
                btnMaxColorDialog.BackColor = colorDialogBox.Color;
            }  
        }

        private void btnMinColorDialog_Click(object sender, EventArgs e)
        {
            if(colorDialogBox.ShowDialog() == DialogResult.OK)  
            {  
                PixelColorValue_t pcv = new PixelColorValue_t();
                pcv.colorFormat = ImageColorFormat_e.RGB;
                pcv.Red = colorDialogBox.Color.R;
                pcv.Green = colorDialogBox.Color.G;
                pcv.Blue = colorDialogBox.Color.B;
                _minColor = pcv.ChangeFormat(_minColor.colorFormat);
                displayColorVals();
                btnMinColorDialog.BackColor = colorDialogBox.Color;
            }  
        }

        private void bnDone_Click(object sender, EventArgs e)
        {
            readColorVals();
            _colorsUpdated = true;
            this.Close();
        }

        private void bnCancel_Click(object sender, EventArgs e)
        {
            _colorsUpdated = false;
            this.Close();
        }

        private void cbColorFormat_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (!_ignoreColorFmtChange)
            {
                ImageColorFormat_e newColorFormat = (ImageColorFormat_e)cbColorFormat.SelectedIndex;
                //Get the current color values
                readColorVals();
                //Display the values in the new format.
                setColorFormat(newColorFormat);
                _ignoreColorFmtChange = false;
            }
        }
    }
}
