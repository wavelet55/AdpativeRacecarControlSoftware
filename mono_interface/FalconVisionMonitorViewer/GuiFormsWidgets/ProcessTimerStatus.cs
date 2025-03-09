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
    public partial class ProcessTimerStatus : UserControl
    {
        public VisionCmdProcess VisionCmdProc;

        public ProcessTimerStatus()
        {
            InitializeComponent();

            cbTimer1Units.Items.Add("sec");
            cbTimer1Units.Items.Add("msec");
            cbTimer1Units.Items.Add("usec");
            cbTimer1Units.SelectedIndex = 0;

            cbTimer2Units.Items.Add("sec");
            cbTimer2Units.Items.Add("msec");
            cbTimer2Units.Items.Add("usec");
            cbTimer2Units.SelectedIndex = 0;
        }

        public void ProcessImageFeatureMatchStatusMsg(FeatureMatchProcStatusPBMsg msg)
        {
            double t1 = msg.ProcessTimer_1;
            if (cbTimer1Units.SelectedIndex == 1) t1 *= 1000.0;
            else if (cbTimer1Units.SelectedIndex == 2) t1 *= 1000000.0;
            tbTimer1.Text = t1.ToString("0.000");

            double t2 = msg.ProcessTimer_2;
            if (cbTimer2Units.SelectedIndex == 1) t2 *= 1000.0;
            else if (cbTimer2Units.SelectedIndex == 2) t2 *= 1000000.0;
            tbTimer2.Text = t2.ToString("0.000");


        }
    }
}
