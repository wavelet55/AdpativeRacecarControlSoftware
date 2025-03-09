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
    public partial class CameraOrientationControl : UserControl
    {
        double _elevationDegrees = 0.0;
        double _azimuthDegrees = 0.0;

        bool _ignoreChange = false;

        public VisionCmdProcess VisionCmdProc;

        public CameraOrientationControl()
        {
            InitializeComponent();
            trackBarElevation.Minimum = -90;
            trackBarElevation.Maximum = 90;
            trackBarElevation.Value = 0;

            trackBarAzimuth.Minimum = -180;
            trackBarAzimuth.Maximum = 180;
            trackBarAzimuth.Value = 0;

        }

        private void btnSend_Click(object sender, EventArgs e)
        {
            _ignoreChange = false;
            //ensure we have the latest values
            tbCElevation_TextChanged(null, null);
            tbAzimuth_TextChanged(null, null);

            CameraSteeringMsg coMsg = new CameraSteeringMsg();
            coMsg.CameraElevationAngleRad = (Math.PI / 180.0) * _elevationDegrees;
            coMsg.CameraAzimuthAngleRad = (Math.PI / 180.0) * _azimuthDegrees;
            coMsg.CameraSteeringModeSPOI = false;

            VisionCmdProc.SendCameraOrientationMsgOnCmdPort(coMsg);
        }

        private void trackBarElevation_Scroll(object sender, EventArgs e)
        {
            _ignoreChange = true;
            _elevationDegrees = trackBarElevation.Value;
            tbCElevation.Text = _elevationDegrees.ToString("0.000");
            _ignoreChange = false;
        }

        private void trackBarAzimuth_Scroll(object sender, EventArgs e)
        {
            _ignoreChange = true;
            _azimuthDegrees = trackBarAzimuth.Value;
            tbAzimuth.Text = _azimuthDegrees.ToString("0.000");
            _ignoreChange = false;
        }

        private void tbCElevation_TextChanged(object sender, EventArgs e)
        {
            if(!_ignoreChange )
            {
                double.TryParse(tbCElevation.Text, out _elevationDegrees);
                _elevationDegrees = _elevationDegrees < -90.0 ? -90.0 : _elevationDegrees > 90.0 ? 90.0 : _elevationDegrees;
                _ignoreChange = true;
                //tbCElevation.Text = _elevationDegrees.ToString("0.000");
                trackBarElevation.Value = (int)(_elevationDegrees);
            }
            _ignoreChange = false;
        }

        private void tbAzimuth_TextChanged(object sender, EventArgs e)
        {
            if(!_ignoreChange )
            {
                double.TryParse(tbAzimuth.Text, out _azimuthDegrees);
                _azimuthDegrees = _azimuthDegrees < -180.0 ? -180.0 : _azimuthDegrees > 180.0 ? 180.0 : _azimuthDegrees;
                _ignoreChange = true;
                //tbAzimuth.Text = _azimuthDegrees.ToString("0.000");
                trackBarAzimuth.Value = (int)(_azimuthDegrees);
            }
            _ignoreChange = false;
        }
    }
}
