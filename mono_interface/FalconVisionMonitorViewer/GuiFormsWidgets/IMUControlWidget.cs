using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using FalconVisionMonitorViewer;
using VisionBridge.Messages;


namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    public partial class IMUControlWidget : UserControl
    {
        public VisionCmdProcess VisionCmdProc;

        bool ignoreChange = false;

        public IMUControlWidget()
        {
            InitializeComponent();
        }

        private void btnSendCmd_Click(object sender, EventArgs e)
        {
            IMUCommandResponsePBMsg cmdMsg = new IMUCommandResponsePBMsg();
            tbCmdResponseMsg.Text = "";
            cmdMsg.IMURemoteCtrlEnable = chkbxRemoteCtrlEnable.Checked;
            cmdMsg.CmdRspMsg = tbIMUSerialCmd.Text;
            VisionCmdProc.SendIMUControlMsgOnCmdPort(cmdMsg);
        }

        public void processIMUCmdResponseMsg(IMUCommandResponsePBMsg rspMsg)
        {
            tbCmdResponseMsg.Text = rspMsg.CmdRspMsg;
            ignoreChange = true;
            chkbxRemoteCtrlEnable.Checked = rspMsg.IMURemoteCtrlEnable;
            ignoreChange = false;
        }

        private void chkbxRemoteCtrlEnable_CheckedChanged(object sender, EventArgs e)
        {
            if (!ignoreChange)
            {
                IMUCommandResponsePBMsg cmdMsg = new IMUCommandResponsePBMsg();
                tbCmdResponseMsg.Text = "";
                cmdMsg.IMURemoteCtrlEnable = chkbxRemoteCtrlEnable.Checked;
                cmdMsg.CmdRspMsg = "";
                VisionCmdProc.SendIMUControlMsgOnCmdPort(cmdMsg);
            }
            ignoreChange = false;
        }
    }
}
