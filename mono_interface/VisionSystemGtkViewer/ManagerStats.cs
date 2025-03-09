using System;
using VisionBridge;
using VisionBridge.Messages;


namespace VisionSystemGtkViewer
{
	[System.ComponentModel.ToolboxItem(true)]
	public partial class ManagerStats : Gtk.Bin
	{
		public string ManagerName;
		public VisionCmdProcess VisionCmdProc;

		public ManagerStats()
		{
			this.Build();
		}

		protected void OnBtnResetMgrStatsButtonPressEvent (object o, Gtk.ButtonPressEventArgs args)
		{
			if (VisionCmdProc != null)
			{
				VisionCmdProc.ManagerControlCmd(ManagerName, true);
			}
		}

		public void SetMgrStats(ManagerStatsMsg mgrStats)
		{
			if (mgrStats != null)
			{
				try
				{
				tbMgrName.Buffer.Text = mgrStats.ManagerName;
				tbMgrState.Buffer.Text = mgrStats.RunningState.ToString();
				tbMgrErrorCode.Buffer.Text = mgrStats.ErrorCode.ToString();
				tbMgrNoExecCycles.Buffer.Text = mgrStats.NumberOfExecuteCycles.ToString();
				tbMgrTime.Buffer.Text = mgrStats.TimeSinceLastStatsReset_Sec.ToString("0000.0");
				tbMgrWakupCallsAwake.Buffer.Text = mgrStats.NumberOfWakeupCallsWhileAwake.ToString();
				tbMgrWakupCallsAsleep.Buffer.Text = mgrStats.NumberOfWakeupCallsWhileAsleep.ToString();
				tbMgrExecTimeMin.Buffer.Text = mgrStats.MinExecuteUnitOfWorkTime_Sec.ToString("0.0000");
				tbMgrExecTimeMax.Buffer.Text = mgrStats.MaxExecuteUnitOfWorkTime_Sec.ToString("0.0000");
				tbMgrExecTimeAve.Buffer.Text = mgrStats.AveExecuteUnitOfWorkTime_Sec.ToString("0.0000");
				tbMgrExecTimeTotal.Buffer.Text = mgrStats.TotalExecuteUnitOfWorkTime_Sec.ToString("0.000");

				tbMgrSleepTimeMin.Buffer.Text = mgrStats.MinSleepTime_Sec.ToString("0.0000");
				tbMgrSleepTimeMax.Buffer.Text = mgrStats.MaxSleepTime_Sec.ToString("0.0000");
				tbMgrSleepTimeAve.Buffer.Text = mgrStats.AveSleepTime_Sec.ToString("0.0000");
				tbMgrSleepTimeTotal.Buffer.Text = mgrStats.TotalSleepTime_Sec.ToString("0.000");
				}
				catch(Exception ex)
				{
					Console.Write("Exeception: " + ex.Message);
				}
			}
		}

		protected void OnBtnResetMgrStatsClicked (object sender, EventArgs e)
		{
			if (VisionCmdProc != null)
			{
				VisionCmdProc.ManagerControlCmd(ManagerName, true, 2.5);
			}
		}
	}

}

