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
    public partial class HeadOrientationSPWidget : UserControl
    {
        public double MaxFwrBackHeadAngleDeg = 60.0;
        public double MaxTiltLeftRightAngleDeg = 45.0;
        public double MaxRotateLeftRightAngleDeg = 90.0;

        public HeadOrientationSPWidget()
        {
            InitializeComponent();
            vScrollBar_HeadFrontBackAngle.Maximum = (int)MaxFwrBackHeadAngleDeg;
            vScrollBar_HeadFrontBackAngle.Minimum = -(int)MaxFwrBackHeadAngleDeg;

            hScrollBar_HeadTiltLRAngle.Maximum = (int)MaxTiltLeftRightAngleDeg;
            hScrollBar_HeadTiltLRAngle.Minimum = -(int)MaxTiltLeftRightAngleDeg;

            hScrollBar_HeadRotationLRAngle.Maximum = (int)MaxRotateLeftRightAngleDeg;
            hScrollBar_HeadRotationLRAngle.Minimum = -(int)MaxRotateLeftRightAngleDeg;
        }

        private void label4_Click(object sender, EventArgs e)
        {

        }

        public void processHeadOrientationMsg(HeadOrientationPBMsg msg)
        {
            double maxAngle;
            tbHeadFrontBackAngle.Text = msg.HeadPitchDegrees.ToString("0.00");
            maxAngle = -msg.HeadPitchDegrees;
            maxAngle = maxAngle < -MaxFwrBackHeadAngleDeg ? -MaxFwrBackHeadAngleDeg : maxAngle > MaxFwrBackHeadAngleDeg ? MaxFwrBackHeadAngleDeg : maxAngle;
            vScrollBar_HeadFrontBackAngle.Value = (int)maxAngle;

            tbHeadTiltLRAngle.Text = msg.HeadRollDegrees.ToString("0.00");
            maxAngle = msg.HeadRollDegrees;
            maxAngle = maxAngle < -MaxTiltLeftRightAngleDeg ? -MaxTiltLeftRightAngleDeg : maxAngle > MaxTiltLeftRightAngleDeg ? MaxTiltLeftRightAngleDeg : maxAngle;
            hScrollBar_HeadTiltLRAngle.Value = (int)maxAngle;

            tbHeadRotationLRAngle.Text = msg.HeadYawDegrees.ToString("0.00");
            maxAngle = -msg.HeadYawDegrees;
            maxAngle = maxAngle < -MaxRotateLeftRightAngleDeg ? -MaxRotateLeftRightAngleDeg : maxAngle > MaxRotateLeftRightAngleDeg ? MaxRotateLeftRightAngleDeg : maxAngle;
            hScrollBar_HeadRotationLRAngle.Value = (int)maxAngle;
        }

        public void processSipAndPuffMsg(SipAndPuffPBMsg msg)
        {
            double maxVal;
            tbSipPuffVal.Text = msg.SipAndPuffPercent.ToString("0.0");
            maxVal = msg.SipAndPuffPercent;
            maxVal = maxVal < -100.0 ? -100.0 : maxVal > 100.0 ? 100.0 : maxVal;
            hScrollBar_SipPuffVal.Value = (int)maxVal;

            tbSipPuffTotalVal.Text = msg.SipAndPuffIntegralPercent.ToString("0.00");
            maxVal = msg.SipAndPuffIntegralPercent;
            maxVal = maxVal < -100.0 ? -100.0 : maxVal > 100.0 ? 100.0 : maxVal;
            hScrollBar_SipPuffTotalVal.Value = (int)maxVal;
        }
    }
}
