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
using GeoCoordinateSystemNS;

namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    public partial class UAVInertialStates : UserControl
    {
        private bool _displayOrSendInertialStates = true;

        public bool DisplayInertialStatesEnabled = false;

        public VisionCmdProcess VisionCmdProc;

        public GeoCoordinateSystem GeoCoordinateSys = null;

        public UAVInertialStates()
        {
            _displayOrSendInertialStates = true;
            //btnSendVIS_LatLon.Text = "Enable";
            //btnSendVIS_LatLon.BackColor = System.Drawing.Color.LightGray;
            //btnSendVIS_LatLon.Visible = true;
            //btnSendVIS_XY.Visible = false;
            DisplayInertialStatesEnabled = false;
            InitializeComponent();
        }

        public void SetDisplayOrSendFormType(bool displayOrSend)
        {
            _displayOrSendInertialStates = displayOrSend;
            if (displayOrSend)
            {
                btnSendVIS_LatLon.Text = "Enable";
                btnSendVIS_LatLon.BackColor = System.Drawing.Color.LightGray;
                btnSendVIS_LatLon.Visible = true;
                btnSendVIS_XY.Visible = false;
                DisplayInertialStatesEnabled = false;
            }
            else
            {
                btnSendVIS_LatLon.Text = "Send LL States";
                btnSendVIS_LatLon.BackColor = System.Drawing.Color.LightGray;
                btnSendVIS_LatLon.Visible = true;

                btnSendVIS_XY.Text = "Send XY States";
                btnSendVIS_XY.BackColor = System.Drawing.Color.LightGray;
                btnSendVIS_XY.Visible = true;
            }
         }


        public void SetInertialStates(VehicleInterialStatesMsg inertialStates)
        {
            if (inertialStates.CoordinatesLatLonOrXY)
            {
                tbUavIS_LatDeg.Text = (inertialStates.LatitudeRadOrY * 180.0 / Math.PI).ToString("0.000000");
                tbUavIS_LonDeg.Text = (inertialStates.LongitudeRadOrX * 180.0 / Math.PI).ToString("0.000000");
                tbUavIS_X.Text = "";
                tbUavIS_Y.Text = "";
            }
            else
            {
                tbUavIS_LatDeg.Text = "";
                tbUavIS_LonDeg.Text = "";
                tbUavIS_X.Text = inertialStates.LongitudeRadOrX.ToString("0.00");
                tbUavIS_Y.Text = inertialStates.LatitudeRadOrY.ToString("0.00");
            }
            tbUavIS_AltMSL.Text = inertialStates.AltitudeMSL.ToString("0.00");
            tbUavIS_HeightAGL.Text = inertialStates.HeightAGL.ToString("0.00");

            tbUavIS_Vel_X.Text = inertialStates.VelEastMpS.ToString("0.00");
            tbUavIS_Vel_Y.Text = inertialStates.VelNorthMpS.ToString("0.00");
            tbUavIS_Vel_Z.Text = inertialStates.VelDownMpS.ToString("0.00");

            tbUavIS_RollDeg.Text = (inertialStates.RollRad * 180.0 / Math.PI).ToString("0.0");
            tbUavIS_PitchDeg.Text = (inertialStates.PitchRad * 180.0 / Math.PI).ToString("0.0");
            tbUavIS_YawDeg.Text = (inertialStates.YawRad * 180.0 / Math.PI).ToString("0.0");

            tbUavIS_RollRate.Text = (inertialStates.RollRateRadps * 180.0 / Math.PI).ToString("0.0");
            tbUavIS_PitchRate.Text = (inertialStates.PitchRateRadps * 180.0 / Math.PI).ToString("0.0");
            tbUavIS_YawRate.Text = (inertialStates.YawRateRadps * 180.0 / Math.PI).ToString("0.0");
        }

        /// <summary>
        /// Read the Inertial States from the text boxes.  If LatLonOrXY is true
        /// uses values from the Lat/Lon position boxes, otherwise uses the values
        /// from the X-Y position boxes.
        /// Returns the Inertial States... or null if there was an error parsing 
        /// a value.
        /// </summary>
        /// <param name="LatLonOrXY"></param>
        /// <returns></returns>
        public VehicleInterialStatesMsg ReadInertialStates(bool LatLonOrXY)
        {
            VehicleInterialStatesMsg inertialStates = new VehicleInterialStatesMsg();
            double value = 0;
            bool parseOK = true;
            if (LatLonOrXY)
            {
                inertialStates.CoordinatesLatLonOrXY = true;
                parseOK &= double.TryParse(tbUavIS_LatDeg.Text, out value);
                inertialStates.LatitudeRadOrY = (Math.PI / 180.0) * value;

                parseOK &= double.TryParse(tbUavIS_LonDeg.Text, out value);
                inertialStates.LongitudeRadOrX = (Math.PI / 180.0) * value;
            }
            else
            {
                inertialStates.CoordinatesLatLonOrXY = false;
                parseOK &= double.TryParse(tbUavIS_Y.Text, out value);
                inertialStates.LatitudeRadOrY = value;

                parseOK &= double.TryParse(tbUavIS_X.Text, out value);
                inertialStates.LongitudeRadOrX = value;
            }

            parseOK &= double.TryParse(tbUavIS_AltMSL.Text, out value);
            inertialStates.AltitudeMSL = value;
            parseOK &= double.TryParse(tbUavIS_HeightAGL.Text, out value);
            inertialStates.HeightAGL = value;

            parseOK &= double.TryParse(tbUavIS_Vel_X.Text, out value);
            inertialStates.VelEastMpS = value;
            parseOK &= double.TryParse(tbUavIS_Vel_Y.Text, out value);
            inertialStates.VelNorthMpS = value;
            parseOK &= double.TryParse(tbUavIS_Vel_Z.Text, out value);
            inertialStates.VelDownMpS = value;

            parseOK &= double.TryParse(tbUavIS_RollDeg.Text, out value);
            inertialStates.RollRad = (Math.PI / 180.0) * value;
            parseOK &= double.TryParse(tbUavIS_PitchDeg.Text, out value);
            inertialStates.PitchRad = (Math.PI / 180.0) * value;
            parseOK &= double.TryParse(tbUavIS_YawDeg.Text, out value);
            inertialStates.YawRad = (Math.PI / 180.0) * value;

            parseOK &= double.TryParse(tbUavIS_RollRate.Text, out value);
            inertialStates.RollRateRadps = (Math.PI / 180.0) * value;
            parseOK &= double.TryParse(tbUavIS_PitchRate.Text, out value);
            inertialStates.PitchRateRadps = (Math.PI / 180.0) * value;
            parseOK &= double.TryParse(tbUavIS_YawRate.Text, out value);
            inertialStates.YawRateRadps = (Math.PI / 180.0) * value;

            if (!parseOK)
            {
                MessageBox.Show("Error Parsing one of the Values... make sure all values are numbers.");
                inertialStates = null;
            }
            return inertialStates;
        }


        //Send or Enable/Disable Display
        private void btnSendVIS_LatLon_Click(object sender, EventArgs e)
        {
            if (_displayOrSendInertialStates)
            {
                DisplayInertialStatesEnabled = !DisplayInertialStatesEnabled;
                if (DisplayInertialStatesEnabled)
                {
                    btnSendVIS_LatLon.Text = "Disable";
                    btnSendVIS_LatLon.BackColor = System.Drawing.Color.Green;
                }
                else
                {
                    btnSendVIS_LatLon.Text = "Enable";
                    btnSendVIS_LatLon.BackColor = System.Drawing.Color.LightGray;
                }
            }
            else
            {
                VehicleInterialStatesMsg inertialStates = ReadInertialStates(true);
                if (inertialStates != null)
                {
                    VisionCmdProc.SendVehicleInertialStatesMsgOnCmdPort(inertialStates);
                }
            }
        }

        private void btnSendVIS_XY_Click(object sender, EventArgs e)
        {
            if (!_displayOrSendInertialStates)
            {
                VehicleInterialStatesMsg inertialStates = ReadInertialStates(false);
                if (inertialStates != null)
                {
                    VisionCmdProc.SendVehicleInertialStatesMsgOnCmdPort(inertialStates);
                }
            }
        }
    }
}
