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
    public partial class VehicleAndImageLocation : UserControl
    {
        private bool _freezeUpdate = false;
        public bool FreezeUpdate
        {
            get { return _freezeUpdate; }
            set
            {
                _freezeUpdate = value;
                freezeUpdateBtnDisplay();
            }
        }

        private GeoCoordinateSystem _geoCoordinateSys = null;
        public GeoCoordinateSystem GeoCoordinateSys
        {
            get { return _geoCoordinateSys; }
            set
            {
                _geoCoordinateSys = value;
                targetLocation_1.GeoCoordinateSys = value;
                targetLocation_2.GeoCoordinateSys = value;
                targetLocation_3.GeoCoordinateSys = value;
                targetLocation_4.GeoCoordinateSys = value;
            }
        }

        public VehicleAndImageLocation()
        {
            InitializeComponent();
            freezeUpdateBtnDisplay();
            targetLocation_1.SetBoxTitle ("Target Number 1");
            targetLocation_2.SetBoxTitle ("Target Number 2");
            targetLocation_3.SetBoxTitle ("Target Number 3");
            targetLocation_4.SetBoxTitle ("Target Number 4");
        }

        private void freezeUpdateBtnDisplay()
        {
            if (FreezeUpdate)
            {
                btnFreezeUpdate.Text = "Frozen";
                btnFreezeUpdate.BackColor = System.Drawing.Color.Red;
            }
            else
            {
                btnFreezeUpdate.Text = "Run";
                btnFreezeUpdate.BackColor = System.Drawing.Color.Green;
            }
        }

        private void btnFreezeUpdate_Click(object sender, EventArgs e)
        {
            FreezeUpdate = !FreezeUpdate;
            freezeUpdateBtnDisplay();
        }

        public void Clear()
        {
            tbImageNumber.Text = "";
            tbVLat.Text = "";
            tbVLon.Text = "";
            tbAltMSL.Text = "";
            tbVelEW.Text = "";
            tbVelNS.Text = "";
            tbVRoll.Text = "";
            tbVPitch.Text = "";
            tbVYaw.Text = "";
        }

        private string ToDegrees5DP(double rad)
        {
            double deg = (180.0 / Math.PI) * rad;
            return deg.ToString("0.00000");
        }

        private string ToDegrees2DP(double rad)
        {
            double deg = (180.0 / Math.PI) * rad;
            return deg.ToString("0.00");
        }

        public void UpdateLocationAndTargetInfo(ImageProcTargetInfoResultsMsg msg)
        {
            if (!_freezeUpdate)
            {
                tbImageNumber.Text = msg.ImageLocation.ImageNumber.ToString();
                if (cbLatLonXYPos.Checked)
                {
                    tbVLat.Text = ToDegrees5DP(msg.VehicleInertialStates.LatitudeRadOrY);
                    tbVLon.Text = ToDegrees5DP(msg.VehicleInertialStates.LongitudeRadOrX);
                }
                else
                {
                    LatLonAltCoord_t latLon = new LatLonAltCoord_t(msg.VehicleInertialStates.LatitudeRadOrY,
                                                                    msg.VehicleInertialStates.LongitudeRadOrX);
                    xyzCoord_t xyPos = GeoCoordinateSys.LatLonAltToXYZ(latLon);

                    tbVLat.Text = xyPos.x.ToString("0.000");
                    tbVLon.Text = xyPos.y.ToString("0.000");
                }
                tbAltMSL.Text = msg.VehicleInertialStates.AltitudeMSL.ToString("0.000");
                tbVelEW.Text = msg.VehicleInertialStates.VelEastMpS.ToString("0.000");
                tbVelNS.Text = msg.VehicleInertialStates.VelNorthMpS.ToString("0.000");
                tbVRoll.Text = ToDegrees2DP( msg.VehicleInertialStates.RollRad);
                tbVPitch.Text = ToDegrees2DP( msg.VehicleInertialStates.PitchRad);
                tbVYaw.Text = ToDegrees2DP( msg.VehicleInertialStates.YawRad);

                if (msg.TargetLocations != null && msg.TargetLocations.Count > 0)
                {
                    if (msg.TargetLocations.Count >= 1)
                    {
                        targetLocation_1.SetTargetInfo(msg.TargetLocations[0]);
                    }
                    else
                    {
                        targetLocation_1.Clear();
                    }
                    if (msg.TargetLocations.Count >= 2)
                    {
                        targetLocation_2.SetTargetInfo(msg.TargetLocations[1]);
                    }
                    else
                    {
                        targetLocation_2.Clear();
                    }
                    if (msg.TargetLocations.Count >= 3)
                    {
                        targetLocation_3.SetTargetInfo(msg.TargetLocations[2]);
                    }
                    else
                    {
                        targetLocation_3.Clear();
                    }
                    if (msg.TargetLocations.Count >= 4)
                    {
                        targetLocation_4.SetTargetInfo(msg.TargetLocations[3]);
                    }
                    else
                    {
                        targetLocation_4.Clear();
                    }
                }
                else
                {
                    targetLocation_1.Clear();
                    targetLocation_2.Clear();
                    targetLocation_3.Clear();
                    targetLocation_4.Clear();
                }

            }

        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            if (cbLatLonXYPos.Checked)
            {
                cbLatLonXYPos.Text = "Lat/Lon";
                lblLatOrXPos.Text = "Lat (Degrees)";
                lblLatOrXPos.Text = "Lon (Degrees)";               
            }
            else
            {
                cbLatLonXYPos.Text = "X / Y";
                lblLatOrXPos.Text = "X (East m)";
                lblLatOrXPos.Text = "Y (North m)";
            }
        }

    }
}
