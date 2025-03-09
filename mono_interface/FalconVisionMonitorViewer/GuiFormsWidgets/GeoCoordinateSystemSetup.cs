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
    public partial class GeoCoordinateSystemSetup : UserControl
    {
        public VisionCmdProcess VisionCmdProc;

        public GeoCoordinateSystem GeoCoordinateSys = null;

        public GeoCoordinateSystemSetup()
        {
            InitializeComponent();
            cbxGeoCConvType.Items.Add("Linear");
            cbxGeoCConvType.Items.Add("WGS84_Relative");
            cbxGeoCConvType.Items.Add("WGS84_Map");
            cbxGeoCConvType.SelectedIndex = 0;
        }


        private void btnSetGeoCoords_Click(object sender, EventArgs e)
        {
            double clat, clon, gAlt;
            bool parseOk = true;
            LatLonAltCoord_t latLon = new LatLonAltCoord_t();
			if (VisionCmdProc != null)
			{
                parseOk &= double.TryParse(tbGCS_CenterLat.Text, out clat);
                parseOk &= double.TryParse(tbGCS_CenterLon.Text, out clon);
                parseOk &= double.TryParse(tbGCS_GroundAltMSL.Text, out gAlt);

                parseOk &= clat >= -90.0 && clat < 90.0;
                parseOk &= clon >= -180.0 && clon < 180.0;
                parseOk &= gAlt >= -1000.0 && gAlt < 10000.0;
                int convType = cbxGeoCConvType.SelectedIndex;
                if (parseOk)
                {
                    latLon.LatitudeDegrees = clat;
                    latLon.LongitudeDegrees = clon;
                    latLon.Altitude = gAlt;
                    //Setup local GeoCoordinateSystem
                    GeoCoordinateSys.SetupGeoCoordinateSystem(latLon, (GeoCoordinateSystemConversionType_e)convType);

                    //Send command to Videre to setup the GeoCoordinateSystem;
                    string respStr = VisionCmdProc.SetupGeoCoordCmd(clat, clon, gAlt, convType);
                    if (respStr.StartsWith("OK"))
                    {
                        btnSetGeoCoords.BackColor = System.Drawing.Color.Green;
                    }
                    else
                    {
                        MessageBox.Show("Error Seting up GeoCoordinate System: " + respStr);
                        btnSetGeoCoords.BackColor = System.Drawing.Color.Red;
                    }
                }   
                else
                {
                    MessageBox.Show("Invalid Paremeter");
                }
			}
        }

        private void btnLatLonToXYConv_Click(object sender, EventArgs e)
        {
            double clat, clon;
            bool parseOk = true;
            if (VisionCmdProc != null)
            {
                parseOk &= double.TryParse(tb_GCSC_Lat.Text, out clat);
                parseOk &= double.TryParse(tb_GCSC_Lon.Text, out clon);
                parseOk &= clat >= -90.0 && clat < 90.0;
                parseOk &= clon >= -180.0 && clon < 180.0;
                if (parseOk)
                {
                    LatLonXYConversionMsg latLonXYConvMsg = VisionCmdProc.GeoCoordLatLonXYConvCmd(true, clat, clon);
                    if (latLonXYConvMsg != null)
                    {
                        tb_GCSC_X.Text = latLonXYConvMsg.X_PosMeters.ToString("0.000");
                        tb_GCSC_Y.Text = latLonXYConvMsg.Y_PosMeters.ToString("0.000");
                    }
                    else
                    {
                        MessageBox.Show("Error getting Lat/Lon to XY Conversion.");
                    }
                }
                else
                {
                    MessageBox.Show("Invalid Paremeter");
                }
            }
        }

        private void btnXYToLatLonConv_Click(object sender, EventArgs e)
        {
            double x, y;
            bool parseOk = true;
            if (VisionCmdProc != null)
            {
                parseOk &= double.TryParse(tb_GCSC_X.Text, out x);
                parseOk &= double.TryParse(tb_GCSC_Y.Text, out y);
                if (parseOk)
                {
                    LatLonXYConversionMsg latLonXYConvMsg = VisionCmdProc.GeoCoordLatLonXYConvCmd(false, x, y);
                    if (latLonXYConvMsg != null)
                    {
                        tb_GCSC_Lat.Text = latLonXYConvMsg.LatitudeDegrees.ToString("0.000000");
                        tb_GCSC_Lon.Text = latLonXYConvMsg.LongitudeDegrees.ToString("0.000000");
                    }
                    else
                    {
                        MessageBox.Show("Error getting Lat/Lon to XY Conversion.");
                    }
                }
                else
                {
                    MessageBox.Show("Invalid Paremeter");
                }
            }
        }

        private void grpbxGeoCoordSetup_Enter(object sender, EventArgs e)
        {

        }

        private void btn_GetGeoCoordSetup_Click(object sender, EventArgs e)
        {
            LatLonAltCoord_t latLon = new LatLonAltCoord_t();
            if (VisionCmdProc != null)
            {
                GeoCoordinateSystemSetupMsg geoCSetupMsg = VisionCmdProc.GetGeoCoordSetupCmd();
                if (geoCSetupMsg != null)
                {
                    tbGCS_CenterLat.Text = geoCSetupMsg.CenterLatitudeDegrees.ToString("0.000000");
                    tbGCS_CenterLon.Text = geoCSetupMsg.CenterLongitudeDegrees.ToString("0.000000");
                    tbGCS_GroundAltMSL.Text = geoCSetupMsg.GroundAltitudeMSL.ToString("0.000");
                    if (geoCSetupMsg.GeoCoordinateSystemConversionType == GeoCoordinateSystemSetupMsg.GeoCoordinateSystemConversionType_e.Linear)
                    {
                        cbxGeoCConvType.SelectedIndex = 0;
                    }
                    else if (geoCSetupMsg.GeoCoordinateSystemConversionType == GeoCoordinateSystemSetupMsg.GeoCoordinateSystemConversionType_e.WGS84_Relative)
                    {
                        cbxGeoCConvType.SelectedIndex = 1;
                    }
                    else if (geoCSetupMsg.GeoCoordinateSystemConversionType == GeoCoordinateSystemSetupMsg.GeoCoordinateSystemConversionType_e.WGS84_Map)
                    {
                        cbxGeoCConvType.SelectedIndex = 2;
                    }

                    //Set the local GeoCoordinate System to the UAV's cooridinate system.
                    latLon.LatitudeDegrees = geoCSetupMsg.CenterLatitudeDegrees;
                    latLon.LongitudeDegrees = geoCSetupMsg.CenterLongitudeDegrees;
                    latLon.Altitude = geoCSetupMsg.GroundAltitudeMSL;
                    //Setup local GeoCoordinateSystem
                    GeoCoordinateSys.SetupGeoCoordinateSystem(latLon, 
                        (GeoCoordinateSystemConversionType_e)geoCSetupMsg.GeoCoordinateSystemConversionType);

                }
                else
                {
                    MessageBox.Show("Error Getting GeoCoordinate System Setup.");
                }
            }
        }
    }
}
