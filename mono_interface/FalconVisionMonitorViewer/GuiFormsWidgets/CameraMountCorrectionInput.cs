using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    public partial class CameraMountCorrectionInput : UserControl
    {

        private double _yawCorrDeg = 0;
        public double YawCorrDeg
        {
            get { return _yawCorrDeg; }
            set { _yawCorrDeg = value < -180 ? -180 : value > 180 ? 180 : value; }
        }

        private double _pitchCorrDeg = 0;
        public double PitchCorrDeg
        {
            get { return _pitchCorrDeg; }
            set { _pitchCorrDeg = value < -180 ? -180 : value > 180 ? 180 : value; }
        }

        private double _rollCorrDeg = 0;
        public double RollCorrDeg
        {
            get { return _rollCorrDeg; }
            set { _rollCorrDeg = value < -180 ? -180 : value > 180 ? 180 : value; }
        }

        private double _delXCorrMilliMeters = 0;
        public double DelXCorrMilliMeters
        {
            get { return _delXCorrMilliMeters; }
            set { _delXCorrMilliMeters = value < -10000 ? -10000 : value > 10000 ? 10000 : value; }
        }

        public double DelXCorrCentiMeters
        {
            get { return 0.1 * _delXCorrMilliMeters; }
            set { DelXCorrMilliMeters = 10 * value; }
        }

        public double DelXCorrInches
        {
            get { return (1.0 / 25.4) * _delXCorrMilliMeters; }
            set { DelXCorrMilliMeters = 25.4 * value; }
        }

        private double _delYCorrMilliMeters = 0;
        public double DelYCorrMilliMeters
        {
            get { return _delYCorrMilliMeters; }
            set { _delYCorrMilliMeters = value < -10000 ? -10000 : value > 10000 ? 10000 : value; }
        }

        public double DelYCorrCentiMeters
        {
            get { return 0.1 * _delYCorrMilliMeters; }
            set { DelYCorrMilliMeters = 10 * value; }
        }

        public double DelYCorrInches
        {
            get { return (1.0 / 25.4) * _delYCorrMilliMeters; }
            set { DelYCorrMilliMeters = 25.4 * value; }
        }

        public CameraMountCorrectionInput()
        {
            InitializeComponent();
            cbMeasUnits.Items.Add("inches");
            cbMeasUnits.Items.Add("mm");
            cbMeasUnits.Items.Add("cm");
            cbMeasUnits.SelectedIndex = 0;

            DisplayValues();
        }

        public void DisplayValues()
        {
            tbYawAngleDeg.Text = YawCorrDeg.ToString("0.000");
            tbPitchAngleDeg.Text = PitchCorrDeg.ToString("0.000");
            tbRollAngleDeg.Text = RollCorrDeg.ToString("0.000");

            int unit = cbMeasUnits.SelectedIndex;
            switch (unit)
            {
                case 0:   //Inches
                    tbDelXPos.Text = DelXCorrInches.ToString("0.000");
                    tbDelYPos.Text = DelYCorrInches.ToString("0.000");
                    break;
                case 1:   //millimeter
                    tbDelXPos.Text = DelXCorrMilliMeters.ToString("0.0");
                    tbDelYPos.Text = DelYCorrMilliMeters.ToString("0.0");
                    break;
                case 2:   //centimeter
                    tbDelXPos.Text = DelXCorrCentiMeters.ToString("0.000");
                    tbDelYPos.Text = DelYCorrCentiMeters.ToString("0.000");
                    break;
            }
        }

        public void ReadValues()
        {
            double value = 0;
            double.TryParse(tbYawAngleDeg.Text, out value);
            YawCorrDeg = value;

            value = 0;
            double.TryParse(tbPitchAngleDeg.Text, out value);
            PitchCorrDeg = value;

            value = 0;
            double.TryParse(tbRollAngleDeg.Text, out value);
            RollCorrDeg = value;

            int unit = cbMeasUnits.SelectedIndex;
            value = 0;
            double.TryParse(tbDelXPos.Text, out value);
            switch (unit)
            {
                case 0:   //Inches
                    DelXCorrInches = value;
                    break;
                case 1:   //millimeter
                    DelXCorrMilliMeters = value;
                    break;
                case 2:   //centimeter
                    DelXCorrCentiMeters = value;
                    break;
            }

            value = 0;
            double.TryParse(tbDelYPos.Text, out value);
            switch (unit)
            {
                case 0:   //Inches
                    DelYCorrInches = value;
                    break;
                case 1:   //millimeter
                    DelYCorrMilliMeters = value;
                    break;
                case 2:   //centimeter
                    DelYCorrCentiMeters = value;
                    break;
            }
        }

        private void btnSet_Click(object sender, EventArgs e)
        {
            //Read and re-display all of the values.
            ReadValues();
            DisplayValues();
        }

        private void btnHelp_Click(object sender, EventArgs e)
        {
            string helpMsg = "Camera Mounting Correction: \n";
            helpMsg += "A camera is assumed mounted to the UAV pointing doww with the top of the image pointed towards the front of the UAV. \n\n";
            helpMsg += "Yaw: a camera with top of the image pointed towards the right wing has a correction of 90.0 degrees.\n\n";

            helpMsg += "Pitch: Correction for camera plane rotated forward (+) or backwards (-).\n\n";

            helpMsg += "Pitch: Correction for camera plane rotated to the right (+) or left (-).\n\n";

            helpMsg += "Del X:  Camera mounted forward (+) or backward (-) of center. \n\n";

            helpMsg += "Del Y:  Camera mounted right (+) or left (-) of center.";

            MessageBox.Show(helpMsg);
        }

        private void cbMeasUnits_SelectedIndexChanged(object sender, EventArgs e)
        {
            DisplayValues();
        }
    }
}
