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
    public partial class GPSFixWidget : UserControl
    {
        public GPSFixWidget()
        {
            InitializeComponent();

            cbUnits.Items.Add("Feet");
            cbUnits.Items.Add("Meters");
            cbUnits.SelectedIndex = 0; ;
        }


        public void processGPSFix(GPSFixPBMsg gpsMsg)
        {
            double metersToFeet = 3.280839895;
            double speed;
            double val;
            tbNoSatelites.Text = gpsMsg.TrackingSatellites.ToString();
            speed = gpsMsg.Velocity_X * gpsMsg.Velocity_X + gpsMsg.Velocity_Y * gpsMsg.Velocity_Y;
            speed = Math.Sqrt(speed);  //Meters / second
            speed = 0.001 * 3600.0 * speed;

            if (cbUnits.SelectedIndex == 0)
            {
                val = metersToFeet * gpsMsg.AltitudeMSL;
                tbAltitude.Text = val.ToString("0.0");

                val = metersToFeet * gpsMsg.Position_X;
                tbPosX.Text = val.ToString("0.0");
                val = metersToFeet * gpsMsg.Position_Y;
                tbPosY.Text = val.ToString("0.0");

                speed = 0.6213711922 * speed;   //To MPH
                tbSpeed.Text = speed.ToString("0.0");
                lblSpeedUnits.Text = "MPH (100 MPH is 100%)";

                int percent = (int)speed;   // 100mph = 100%
                percent = percent < 0 ? 0 : percent > 100 ? 100 : percent;
                pBarSpeed.Value = percent;
            }
            else
            {
                tbAltitude.Text = gpsMsg.AltitudeMSL.ToString("0.0");
                tbPosX.Text = gpsMsg.Position_X.ToString("0.0");
                tbPosY.Text = gpsMsg.Position_Y.ToString("0.0");
                tbSpeed.Text = speed.ToString("0.0");  
                lblSpeedUnits.Text = "KPH  (200 KPH is 100%)";
                int percent = (int)(0.5 * speed);   // 200kph = 100%
                percent = percent < 0 ? 0 : percent > 100 ? 100 : percent;
                pBarSpeed.Value = percent;
            }
        }

    }
}
