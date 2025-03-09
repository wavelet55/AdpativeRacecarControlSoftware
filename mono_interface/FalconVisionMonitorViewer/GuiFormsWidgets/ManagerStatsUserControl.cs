/* ****************************************************************
 * Vision System Viewer
 * DireenTech Inc.  (www.direentech.com)
 * Athrs: Randy Direen PhD
 * 		  Harry Direen PhD
 * Date: Aug. 2016
 * 
 * Developed under contract for:
 * Academy Center for UAS Research 
 * Department of Electrical and Computer Engineering 
 * HQ USAFA/DFEC 
 * 2354 Fairchild Drive 
 * USAF Academy, CO 80840-6236 
 *******************************************************************/

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using VisionBridge;
using VisionBridge.Messages;

namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    public partial class ManagerStatsUserControl : UserControl
    {
        public string ManagerName
        {
            get
            {
                string mgrName = "UnknownMgr";
                if (cbxManagerName.Items.Count > 0)
                {
                    mgrName = (string)cbxManagerName.SelectedItem;
                }
                return mgrName;
            }
        }

		public VisionCmdProcess VisionCmdProc;

        public int MgrIndexNo = 0;

        public ManagerStatsUserControl()
        {
            InitializeComponent();
        }


        public void SetManagerNames(string[] listOfMgrNames)
        {
            if (listOfMgrNames != null && listOfMgrNames.Length > 0)
            {
                cbxManagerName.Items.Clear();
                foreach (string mgrName in listOfMgrNames)
                {
                    cbxManagerName.Items.Add(mgrName);
                }
                if (MgrIndexNo >= 0 && MgrIndexNo < cbxManagerName.Items.Count)
                {
                    cbxManagerName.SelectedIndex = MgrIndexNo;
                }
            }
        }


        private void btnResetStats_Click(object sender, EventArgs e)
        {
            double updateTimeSec = 10.0;
			if (VisionCmdProc != null && cbxManagerName.Items.Count > 0)
			{
                if( !double.TryParse(tbStatsUpdateTimeSec.Text , out updateTimeSec) )
                {
                    updateTimeSec = 10.0;
                    tbStatsUpdateTimeSec.Text = updateTimeSec.ToString();
                }
				VisionCmdProc.ManagerControlCmd(ManagerName, true, updateTimeSec);
			}
        }

		public void SetMgrStats(ManagerStatsMsg mgrStats)
		{
			if (mgrStats != null)
			{
				tbMgrRunningState.Text = mgrStats.RunningState.ToString();
				tbMgrErrorCode.Text = mgrStats.ErrorCode.ToString();
				tbNumExecCycles.Text = mgrStats.NumberOfExecuteCycles.ToString();
				tbMgrExecTimeSec.Text = mgrStats.TimeSinceLastStatsReset_Sec.ToString("0.0");
				tbNoWakeUpsAwake.Text = mgrStats.NumberOfWakeupCallsWhileAwake.ToString();
				tbNoWakeUpsAsleep.Text = mgrStats.NumberOfWakeupCallsWhileAsleep.ToString();

				tbMinExecTimeSec.Text = mgrStats.MinExecuteUnitOfWorkTime_Sec.ToString("0.0000");
				tbMaxExecTimeSec.Text = mgrStats.MaxExecuteUnitOfWorkTime_Sec.ToString("0.0000");
				tbAveExecTimeSec.Text = mgrStats.AveExecuteUnitOfWorkTime_Sec.ToString("0.0000");
				tbTotalExecTimeSec.Text = mgrStats.TotalExecuteUnitOfWorkTime_Sec.ToString("0.000");

				tbMinSleepTimeSec.Text = mgrStats.MinSleepTime_Sec.ToString("0.0000");
				tbMaxSleepTimeSec.Text = mgrStats.MaxSleepTime_Sec.ToString("0.0000");
				tbAveSleepTimeSec.Text = mgrStats.AveSleepTime_Sec.ToString("0.0000");
				tbTotalSleepTimeSec.Text = mgrStats.TotalSleepTime_Sec.ToString("0.000");
			}
		}

        private void cbxManagerName_SelectedIndexChanged(object sender, EventArgs e)
        {
            MgrIndexNo = cbxManagerName.SelectedIndex;
        }

    }
}
