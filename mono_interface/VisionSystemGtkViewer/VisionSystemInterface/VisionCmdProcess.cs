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
using VisionBridge;
using VisionBridge.Messages;

namespace VisionSystemGtkViewer
{
	/// <summary>
	/// This class handles the Command/Response Messages between
	/// the Vision System Viewer and The Vision System
	/// </summary>
	public class VisionCmdProcess
	{
		private Bridge _bridge;

		private int _numberCmdsSent = 0;
		public int NumberCmdsSent
		{
			get { return _numberCmdsSent; }
		}

		private int _numberCmdErrors = 0;
		public int NumberCmdErrors
		{
			get { return _numberCmdErrors; }
		}

		public VisionCmdProcess(Bridge vsbridge)
		{
			_bridge = vsbridge;
		}

		/// <summary>
		/// Send a simple command that is defined by the "cmd" string
		/// with an optional command qualifier. 
		/// </summary>
		/// <returns>The cmd responc string from the Vision System.</returns>
		/// <param name="cmd">Cmd.</param>
		/// <param name="cmdQualifier">Cmd qualifier.</param>
		public string SimpleCmdResponce(string cmd, string cmdQualifier = null)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = cmd };
			if (!string.IsNullOrEmpty(cmdQualifier))
			{
				cmdMsg.CmdQualifier = cmdQualifier;
			}
			VisionResponsePBMsg responseMsg = _bridge.TransmitCommandResponseMessage(cmdMsg);
			if (responseMsg.CmdResponseType == VisionResponsePBMsg.ResponseType_e.ERROR)
			{
				_numberCmdErrors++;
			}
			respStr = responseMsg.CmdResponseType.ToString();
			if (responseMsg.CmdResponseMessage != null)
			{
				respStr += ":" + responseMsg.CmdResponseMessage;
			}
			_numberCmdsSent++;
			return respStr;
		}


		public string ManagerControlCmd(string mgrName, bool resetStatsToggleFlag, double publishMgrStatsTimeSec = 10.0, bool shutdownMgr = false)
		{
			string respStr = "";
			ManagerControlMsg mgrCtrlMsg = new ManagerControlMsg();
			mgrCtrlMsg.ManagerName = mgrName;
			mgrCtrlMsg.ResetManagerStatsToggle = resetStatsToggleFlag;
			mgrCtrlMsg.PublishMgrStatsTime_Sec = publishMgrStatsTimeSec;
			mgrCtrlMsg.ShutdownManager = shutdownMgr;

			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "ManagerControlCmd" };
			cmdMsg.CmdData = mgrCtrlMsg.Serialize();
			VisionResponsePBMsg responseMsg = _bridge.TransmitCommandResponseMessage(cmdMsg);
			if (responseMsg.CmdResponseType == VisionResponsePBMsg.ResponseType_e.ERROR)
			{
				_numberCmdErrors++;
			}
			respStr = responseMsg.CmdResponseType.ToString();
			if (responseMsg.CmdResponseMessage != null)
			{
				respStr += ":" + responseMsg.CmdResponseMessage;
			}
			_numberCmdsSent++;
			return respStr;
		}

		public ManagerStatsMsg GetManagerStats(string mgrName, out string cmdResponseMsg)
		{
			cmdResponseMsg = "Error";
			ManagerStatsMsg mgrStatsMsg = null;
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "GetManagerStats" };
			cmdMsg.CmdQualifier = mgrName;
			VisionResponsePBMsg responseMsg = _bridge.TransmitCommandResponseMessage(cmdMsg);
			cmdResponseMsg = responseMsg.CmdResponseType.ToString();
			cmdResponseMsg += ":" + responseMsg.CmdResponseMessage;
			if (responseMsg.CmdResponseType == VisionResponsePBMsg.ResponseType_e.OK)
			{
				if (responseMsg.CmdResponseData != null)
				{
					try
					{
						mgrStatsMsg = ManagerStatsMsg.Deserialize(responseMsg.CmdResponseData);
					} 
					catch (Exception ex)
					{
						mgrStatsMsg = null;
						cmdResponseMsg = string.Concat(cmdResponseMsg, "::ErrorDeserializing: ", ex.Message);
					}
				}
			} 
			return mgrStatsMsg;
		}

	}
}

