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
    public partial class DisplayImageInfo : UserControl
    {
        public DisplayImageInfo()
        {
            InitializeComponent();
        }

        public void SetImageLocInfo(ImageLocationMsg imgLoc)
        {
            tbImageInfo_INumber.Text = imgLoc.ImageNumber.ToString();

            if (imgLoc.TargetCornerLocations != null && imgLoc.TargetCornerLocations.Count >= 8)
            {
                tbImageInfo_C1x.Text = imgLoc.TargetCornerLocations[0].ToString("0.00");
                tbImageInfo_C1y.Text = imgLoc.TargetCornerLocations[1].ToString("0.00");
                tbImageInfo_C2x.Text = imgLoc.TargetCornerLocations[2].ToString("0.00");
                tbImageInfo_C2y.Text = imgLoc.TargetCornerLocations[3].ToString("0.00");
                tbImageInfo_C3x.Text = imgLoc.TargetCornerLocations[4].ToString("0.00");
                tbImageInfo_C3y.Text = imgLoc.TargetCornerLocations[5].ToString("0.00");
                tbImageInfo_C4x.Text = imgLoc.TargetCornerLocations[6].ToString("0.00");
                tbImageInfo_C4y.Text = imgLoc.TargetCornerLocations[7].ToString("0.00");
            }
        }
    }
}
