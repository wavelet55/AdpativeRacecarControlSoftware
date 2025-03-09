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
    public partial class TargetLocation : UserControl
    {
        public GeoCoordinateSystem GeoCoordinateSys = null;

        public TargetLocation()
        {
            InitializeComponent();
            cbLatLonXYDisp.Checked = false;
        }

        public void SetBoxTitle(string title)
        {
            gboxTargetInfo.Text = title;
        }

        public void Clear()
        {
            btTargetType.Text = "No Target";
            btTargetType.BackColor = System.Drawing.Color.LightGray;
            tbTgtLat.Text = "";
            tbTgtLon.Text = "";
            tbTgtPxlX.Text = "";
            tbTgtPxlY.Text = "";
            tbTgtAzimuth.Text = "";
            tbTgtElevation.Text = "";
            tbTgtOrientationAngle.Text = "";
            tbCv00.Text = "";
            tbCv01.Text = "";
            tbCv10.Text = "";
            tbCv11.Text = "";
        }

        private string ToDegreesStr2DP(double rads)
        {
            double deg =  (180.0 / Math.PI) * rads;
            return deg.ToString("0.00");
        }

        private string ToDegreesStr5DP(double rads)
        {
            double deg =  (180.0 / Math.PI) * rads;
            return deg.ToString("0.00000");
        }

        public void SetTargetInfo(GroundTargetLocationMsg targetInfo)
        {
            if (targetInfo.TargetTypeCode > 0)
            {
                btTargetType.Text = "Tgt Type: " + targetInfo.TargetTypeCode.ToString();
                btTargetType.BackColor = System.Drawing.Color.Green;
                if (cbLatLonXYDisp.Checked)
                {
                    tbTgtLat.Text = ToDegreesStr5DP(targetInfo.TargetLatitudeRadians);
                    tbTgtLon.Text = ToDegreesStr5DP(targetInfo.TargetLongitudeRadians);
                }
                else
                {
                    LatLonAltCoord_t latLon = new LatLonAltCoord_t(targetInfo.TargetLatitudeRadians,
                                                                    targetInfo.TargetLongitudeRadians);
                    xyzCoord_t xyPos = GeoCoordinateSys.LatLonAltToXYZ(latLon);

                    tbTgtLat.Text = xyPos.x.ToString("0.000");
                    tbTgtLon.Text = xyPos.y.ToString("0.000");
                }
                tbTgtPxlX.Text = targetInfo.TargetPixelLocation_x.ToString();
                tbTgtPxlY.Text = targetInfo.TargetPixelLocation_y.ToString();
                tbTgtAzimuth.Text = ToDegreesStr2DP(targetInfo.TargetAzimuthRadians);
                tbTgtElevation.Text = ToDegreesStr2DP(targetInfo.TargetElevationRadians);;
                tbTgtOrientationAngle.Text = ToDegreesStr2DP(targetInfo.TargetOrientationRadians);
                if (targetInfo.TargetCovarianceMatrix != null && targetInfo.TargetCovarianceMatrix.Count >= 4)
                {
                    tbCv00.Text = targetInfo.TargetCovarianceMatrix[0].ToString("0.00");
                    tbCv01.Text = targetInfo.TargetCovarianceMatrix[1].ToString("0.00");
                    tbCv10.Text = targetInfo.TargetCovarianceMatrix[2].ToString("0.00");
                    tbCv11.Text = targetInfo.TargetCovarianceMatrix[3].ToString("0.00");
                }
                else
                {
                    tbCv00.Text = "";
                    tbCv01.Text = "";
                    tbCv10.Text = "";
                    tbCv11.Text = "";
                }
            }
            else
            {
                Clear();
            }
        }

        private void btTargetType_Click(object sender, EventArgs e)
        {
            //Nothing to to to... this button is being used for display purposes
        }

        private void cbLatLonXYDisp_CheckedChanged(object sender, EventArgs e)
        {
            if (cbLatLonXYDisp.Checked)
            {
                cbLatLonXYDisp.Text = "Lat/Lon";
                lblLatLonXY.Text = "Lat/Lon";
            }
            else
            {
                cbLatLonXYDisp.Text = "X / Y";
                lblLatLonXY.Text = "X/Y (m)";
            }
        }
    }
}
