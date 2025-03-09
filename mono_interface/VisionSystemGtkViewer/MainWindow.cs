/* ****************************************************************
 * Vision Bridge
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Randy Direen PhD
 * Date: Aug. 2015
 * 
 * Developed under contract for:
 * Academy Center for UAS Research 
 * Department of Electrical and Computer Engineering 
 * HQ USAFA/DFEC 
 * 2354 Fairchild Drive 
 * USAF Academy, CO 80840-6236 
 *******************************************************************/
using System;
using Gtk;
using VisionBridge;
using VisionBridge.Messages;
using System.Net;
using System.Net.Sockets;
using System.Drawing;
using System.Threading.Tasks;
using System.IO;
using System.ComponentModel;
using System.Drawing.Imaging;
using VisionSystemGtkViewer;


public partial class MainWindow: Gtk.Window
{
	
	private Bridge _bridge;
	private long _numMessages;
	private bool _connectedToVisionBridge = false;

	private VisionCmdProcess _visionCmdProcess;

	public IPEndPoint ServerEndPoint;
	public System.Net.Sockets.Socket WinSocket;
	public IPEndPoint senderUDP;
	public EndPoint Remote;
	public byte[] data;

	public bool sendTelemetryToVisionSystem;

	Gdk.Pixbuf pixbuff1;
	Gdk.Pixbuf pixbuff2;

	public MainWindow()
		: base(Gtk.WindowType.Toplevel)
	{
		_numMessages = 0;
		_bridge = new Bridge();
		_visionCmdProcess = new VisionCmdProcess(_bridge);
		Build();
		tbTcpAddrInput.Buffer.Text = "192.168.1.107";
		tbThisCompTcpAddrInput.Buffer.Text = "192.168.1.105";

		mgrStatsComm.ManagerName = "CommsManager";
		mgrStatsComm.VisionCmdProc = _visionCmdProcess;

		mgrStatsImageCapture.ManagerName = "ImageCaptureManager";
		mgrStatsImageCapture.VisionCmdProc = _visionCmdProcess;

		mgrStatsImageProc.ManagerName = "VisionProcessManager";
		mgrStatsImageProc.VisionCmdProc = _visionCmdProcess;

		mgrStatsSteamRecord.ManagerName = "StreamRecordManager";
		mgrStatsSteamRecord.VisionCmdProc = _visionCmdProcess;

	}

	private string ConnectVisionBridge(string visionSystemTpcAddr,
		string bridgeSystemTcpAddr)
	{
		string errorMsg = null;
		_bridge.VisionSystemConnectionType = BridgeConnectionType_e.tcp;
		_bridge.VisionSystemTCPAddrString = visionSystemTpcAddr;
		_bridge.BridgeSystemTCPAddrString = bridgeSystemTcpAddr;
		_bridge.VisionSystemCommandResponsePort = 5555;
		_bridge.BridgeSubscribeToVisionResultsPort = 5556;
		_bridge.BridgeSubscribeToVisionVideoStreamPort = 5557;

		_bridge.MaxCmdResponseWaitTimeSeconds = 1.5;
		errorMsg = _bridge.ConnectToVisionSystemCommandResponseSocket();
		if (errorMsg != null)
		{
			return errorMsg;
		}

		errorMsg = _bridge.ConnectPublishTelemeteryToVisionSystemSocket();
		if( errorMsg != null )
		{
			_bridge.DisconnectAllSockets();
			return errorMsg;
		}

		sendTelemetryToVisionSystem = true;
		PublishStuffToVisionSystem();

		_bridge.SetActionForRxVisionResultsMessage(DoBackgroundWork);
		_bridge.SetActionForRxVideoStream(DoBackgroundVideoRead);

		_bridge.ConnectSubscribeToVisionResulstSocketAsync();
		_bridge.ConnectSubscribeToVisionSystemVideoStreamSocketAsync();
		return errorMsg;
	}

	//Currently there is no easy way to disconnect the vision bridge 
	//other than to shutdown and restart.
	private string DisconnectVisionBridge()
	{
		string errorMsg = null;
		sendTelemetryToVisionSystem = false;


		return errorMsg;
	}

	private void DoBackgroundWork(string message, byte[] data)
	{
		var pixbuf = new Gdk.Pixbuf(data);
		pixbuff1 = pixbuf.ScaleSimple(567, 320, Gdk.InterpType.Bilinear); 
		
		Gtk.Application.Invoke(delegate {
			if (data.Length > 0)
			{
				image1.Pixbuf = pixbuff1;
			}
			textVisionOutput.Buffer.Text = message;
			_numMessages++;
			//textShowNumPackets.Buffer.Text = _numMessages.ToString();
		});
	}

	private void DoBackgroundVideoRead(byte[] image_data)
	{
		var pixbuf = new Gdk.Pixbuf(image_data);
		pixbuff2 = pixbuf.ScaleSimple(567, 320, Gdk.InterpType.Bilinear); 

		Gtk.Application.Invoke(delegate {
			if (image_data.Length > 0)
			{
				image2.Pixbuf = pixbuff2;
			}
		});

	}

	protected void OnDeleteEvent(object sender, DeleteEventArgs a)
	{
		_bridge.Dispose();
		sendTelemetryToVisionSystem = false;
		Application.Quit();
		a.RetVal = true;
	}

	protected void OnInfoButtonClicked(object sender, EventArgs e)
	{
		TransmitSimpleMessage("Info");
	}

	protected void OnSystemClicked (object sender, EventArgs e)
	{
		TransmitSimpleMessage("System");
	}

	protected void OnStartVideoButtonClicked(object sender, EventArgs e)
	{
		TransmitSimpleMessage("StartVision");
	}

	protected void OnStopVideoButtonClicked(object sender, EventArgs e)
	{
		TransmitSimpleMessage("StopVision");
	}

	protected void OnStartStreamButtonClicked(object sender, EventArgs e)
	{
		TransmitSimpleMessage("StartStream");
	}

	protected void OnStopStreamButtonClicked(object sender, EventArgs e)
	{
		TransmitSimpleMessage("StopStream");
	}
		

	protected void OnGPUStartClicked (object sender, EventArgs e)
	{
		TransmitSimpleMessage("GPUEnable");
	}

	protected void OnGPUStopClicked (object sender, EventArgs e)
	{
		TransmitSimpleMessage("GPUDisable");
	}

	protected void OnStopRecordingClicked (object sender, EventArgs e)
	{
		//TransmitSimpleMessage("StopRecording");
		string resultsMsg = _visionCmdProcess.ManagerControlCmd("ImageCaptureManager", true, 5.5);
		textOutputWindow.Buffer.Text = resultsMsg;
	}

	protected void OnStartRecordingClicked (object sender, EventArgs e)
	{
		string resultsMsg;
		ManagerStatsMsg mgrStats;
		//TransmitSimpleMessage("StartRecording");
		mgrStats = _visionCmdProcess.GetManagerStats("ImageCaptureManager", out resultsMsg);
		textOutputWindow.Buffer.Text = resultsMsg;
	}

	protected void OnKillButtonClicked(object sender, EventArgs e)
	{
		TransmitSimpleMessage("Kill");
	}

	private void TransmitSimpleMessage(string message)
	{
		string respStr = _visionCmdProcess.SimpleCmdResponce(message);
		textOutputWindow.Buffer.Text = respStr;
		//textShowNumPackets.Buffer.Text = _visionCmdProcess.NumberCmdsSent.ToString();
	}

	private async void PublishStuffToVisionSystem()
	{
		int cntr = 0;
		var msg = new VehicleInterialStatesMsg();
		msg.CoordinatesLatLonOrXY = true;
		msg.LatitudeRadOrY = 2.1111111111111111;
		msg.LongitudeRadOrX = 3;
		msg.AltitudeMSL = 4;
		msg.HeightAGL = 5;
		msg.VelEastMpS = 6.1111111111111111;
		msg.VelNorthMpS = 7;
		msg.VelDownMpS = 8;
		msg.RollRad = 9;
		msg.PitchRad = 10.1111111111111111;
		msg.YawRad = 11;
		msg.RollRateRadps = 12;
		msg.PitchRateRadps = 13.1111111111111111;
		msg.YawRateRadps = 14;
		msg.gpsTimeStampSec = 15.1111111111111111;

		while(sendTelemetryToVisionSystem)
		{
			try
			{
				_bridge.PublishTelemetryToVisionSystem(msg);

				//Get Manager Stats... this is a temp location for handling this;
				//*********************
				int statsNo = cntr % 4;
				if (statsNo == 0)
				{
					string resultsMsg;
					ManagerStatsMsg mgrStats = _visionCmdProcess.GetManagerStats("CommsManager", out resultsMsg);
					mgrStatsComm.SetMgrStats(mgrStats);
				} 
				else if (statsNo == 1)
				{
					string resultsMsg;
					ManagerStatsMsg mgrStats = _visionCmdProcess.GetManagerStats("ImageCaptureManager", out resultsMsg);
					mgrStatsImageCapture.SetMgrStats(mgrStats);
				}
				else if (statsNo == 2)
				{
					string resultsMsg;
					ManagerStatsMsg mgrStats = _visionCmdProcess.GetManagerStats("VisionProcessManager", out resultsMsg);
					mgrStatsImageProc.SetMgrStats(mgrStats);
				}
				else if (statsNo == 3)
				{
					string resultsMsg;
					ManagerStatsMsg mgrStats = _visionCmdProcess.GetManagerStats("StreamRecordManager", out resultsMsg);
					mgrStatsSteamRecord.SetMgrStats(mgrStats);
				}
				//**************/
				++cntr;
			}
			catch(Exception ex)
			{
				Console.Write("Exception: " + ex.Message);
			}
			await Task.Delay(1000);
		}
	}

	protected void OnBtnZMQConnectClicked (object sender, EventArgs e)
	{
		if (!_connectedToVisionBridge)
		{
			string visionSystemTcpAddr = tbTcpAddrInput.Buffer.Text;
			string bridgeSystemTcpAddr = tbThisCompTcpAddrInput.Buffer.Text;
			string errorMsg = ConnectVisionBridge(visionSystemTcpAddr, bridgeSystemTcpAddr);
			if (errorMsg != null)
			{
				textOutputWindow.Buffer.Text = errorMsg;
				_connectedToVisionBridge = false;
			} else
			{
				textOutputWindow.Buffer.Text = "Connected To Vision System";
				tbTcpAddrInput.Buffer.Text = "Connected To:" + visionSystemTcpAddr;
				//this.btnZMQConnect.Colormap =;
				_connectedToVisionBridge = true;
			}
		}
	}
}
