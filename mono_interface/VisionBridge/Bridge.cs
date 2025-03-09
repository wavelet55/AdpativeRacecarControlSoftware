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
using NetMQ;
using NetMQ.Sockets;
using ProtoBuf;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using VisionBridge.Messages;


namespace VisionBridge
{
	/// <summary>
	/// The Connection types supported by the Vision System Bridge
	/// </summary>
	public enum BridgeConnectionType_e
	{
		/// <summary>
		/// Standard communications protocol... can be used with the 
		/// Vision System on the same computer or on another computer.
		/// Works on all platforms.
		/// The Connection string is an IP address or localhost.
		/// </summary>
		tcp,
		/// <summary>
		/// Inter-Process Communications... only supported on Linux
		/// Use a filename for the Connection String.
		/// </summary>
		ipc
	}


    /// <summary>
    /// This is the tool for connecting HOPS Framework to the computer vision 
    /// code written in C++.
    /// </summary>
    public class Bridge : IDisposable
    {

		#region BridgeProperties

		private BridgeConnectionType_e _connectionType = BridgeConnectionType_e.tcp;
		public BridgeConnectionType_e VisionSystemConnectionType 
		{
			get { return _connectionType; }
			set 
			{
				_connectionType = value;
				_cmdResponseProcess.BridgeConnectionType = value;
			}
		}

		private string _visionSystemTCPAddrString = "localhost";
		/// <summary>
		/// The TCP Address in the format:
		/// "127.0.0.1" or "localhost"
		/// </summary>
		/// <value>The vision system connection type and address string.</value>
		public string VisionSystemTCPAddrString 
		{
			get { return _visionSystemTCPAddrString; }
			set 
			{ 
				_visionSystemTCPAddrString = string.IsNullOrEmpty(value) ? "localhost" : value.Trim().ToLower(); 
				_cmdResponseProcess.TCPAddrString = _visionSystemTCPAddrString;
                _vsResultsMsgProcessor.VisionSystemTCPAddrString = _visionSystemTCPAddrString;
                _vsVideoStreamMsgProcessor.VisionSystemTCPAddrString = _visionSystemTCPAddrString;
                _vsMonitorMsgProcessor.VisionSystemTCPAddrString = _visionSystemTCPAddrString;
			}
		}

		private string _bridgeSystemTCPAddrString = "localhost";
		/// <summary>
		/// The TCP Address in the format:
		/// "127.0.0.1" or "localhost for the Computer the Bridge is On."
		/// </summary>
		/// <value>The vision system connection type and address string.</value>
		public string BridgeSystemTCPAddrString 
		{
			get { return _bridgeSystemTCPAddrString; }
			set 
			{ 
				_bridgeSystemTCPAddrString = string.IsNullOrEmpty(value) ? "localhost" : value.Trim().ToLower(); 
			}
		}

		/// <summary>
		/// The Vision System Request / Responce Socket Port for tcp connections for tcp connections.
		/// </summary>
		/// <value>The vision system command port.</value>
		public int VisionSystemCommandResponsePort
		{
			get { return _cmdResponseProcess.TCPPort; }
			set { _cmdResponseProcess.TCPPort = value; }
		}


		private int _bridgePublishTelemetryPort = 5558;
		/// <summary>
		/// The Bridge System Publishs Messages To Vision System Socket Port for tcp connections.
		/// </summary>
		/// <value>The vision system command port.</value>
		public int BridgePublishTelemetryPort
		{
			get { return _bridgePublishTelemetryPort; }
			set { _bridgePublishTelemetryPort = value < 1 ? 1 : value > 65535 ? 65535 : value; }
		}

		/// <summary>
		/// The Vision System Command Response Inter-Process Comm Filename
		/// </summary>
		/// <value>The vision system command port.</value>
		public string VisionSystemCmdResponseIPCFilename 
		{
			get { return _cmdResponseProcess.IPCFilename; }
			set { _cmdResponseProcess.IPCFilename = value; }
		}


		private string _bridgePublishTelemetrySocketIPCFilename = "/VisionSystemIPCDir/VSPublishTelemetrySocketIPC";
		/// <summary>
		/// The Publish Messages To Vision System Inter-Process Comm Filename
		/// </summary>
		/// <value>The vision system command port.</value>
		public string BridgePublishTelemetrySocketIPCFilename 
		{
			get { return _bridgePublishTelemetrySocketIPCFilename; }
			set { _bridgePublishTelemetrySocketIPCFilename = string.IsNullOrEmpty(value) ? "/VisionSystemIPCDir/VSPublishTelemetrySocketIPC" : value.Trim(); }
		}

		/// <summary>
		/// Is the Vision System Request Process Socket Connected.
		/// </summary>
		/// <value><c>true</c> if this instance is vision command socket connected; otherwise, <c>false</c>.</value>
        public bool IsCommandResponseSocketConnected 
        { 
			get { return _cmdResponseProcess.IsConnected; }
        }

        //This socket is for publishing  telemetry information such as the Vehicle Intertial 
        //states to the Vision System.
		private PublisherSocket _publishTelemetryToVisionSystemSocket = null;

		/// <summary>
		/// Is the Publish to Vision System Video Socket Connected.
		/// </summary>
		/// <value><c>true</c> if this instance is publish to vision socket connected; otherwise, <c>false</c>.</value>
		public bool IsPublishTelemetryToVisionSocketConnected 
		{ 
			get { return _publishTelemetryToVisionSystemSocket != null ? true : false; }
		}

		/// <summary>
		/// The Maximum time to wait for the Vision System to respond to 
		/// a Request/Command.  If the vision system fails to respond in this
		/// timeframe, and error will be declared and the socket will be reset.
		/// </summary>
		/// <value>The max response wait time seconds.</value>
		public double MaxCmdResponseWaitTimeSeconds 
		{
			get { return _cmdResponseProcess.MaxResponseWaitTimeSeconds; }
			set { _cmdResponseProcess.MaxResponseWaitTimeSeconds = value; }
		}

		#endregion

		//private readonly NetMQContext _zmqContext;

        //Command Responce Process Handler
		private BridgeCommandResponseProcess _cmdResponseProcess = null;

        //Subscription Socket Processors
        private BridgeSubscriptionSocketMessageProcess _vsResultsMsgProcessor;
        /// <summary>
        /// Vision System Results Messsage Subscription Socket
        /// Message processor/handler.  The results of Image Processing
        /// are published on this socket.
        /// </summary>
        public BridgeSubscriptionSocketMessageProcess VSResultsMsgProcessor
        {
            get { return _vsResultsMsgProcessor; }
        }

        private BridgeSubscriptionSocketMessageProcess _vsVideoStreamMsgProcessor;
        /// <summary>
        /// Vision System Video Stream Messsage Subscription Socket
        /// Message processor/handler.
        /// </summary>
        public BridgeSubscriptionSocketMessageProcess VSVideoStreamMsgProcessor
        {
            get { return _vsVideoStreamMsgProcessor; }
        }    
        
        private BridgeSubscriptionSocketMessageProcess _vsMonitorMsgProcessor;
        /// <summary>
        /// Vision System Performance Monitor Messsage Subscription Socket
        /// Message processor/handler.
        /// </summary>
        public BridgeSubscriptionSocketMessageProcess VSMonitorMsgProcessor
        {
            get { return _vsMonitorMsgProcessor; }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="VisionBridge.Bridge"/> class.
        /// </summary>
        public Bridge()
        {
			//_zmqContext = NetMQContext.Create();
			_cmdResponseProcess = new BridgeCommandResponseProcess();

            _vsResultsMsgProcessor = new BridgeSubscriptionSocketMessageProcess();
            _vsResultsMsgProcessor.TCPPort = 5556;
            _vsResultsMsgProcessor.IPCFilename = "/VisionSystemIPCDir/VSResponseMsgSocketIPC";

            _vsVideoStreamMsgProcessor = new BridgeSubscriptionSocketMessageProcess();
            _vsVideoStreamMsgProcessor.TCPPort = 5557;
            _vsVideoStreamMsgProcessor.IPCFilename = "/VisionSystemIPCDir/VSVideoStreamMsgSocketIPC";

            _vsMonitorMsgProcessor = new BridgeSubscriptionSocketMessageProcess();
            _vsMonitorMsgProcessor.TCPPort = 5559;
            _vsMonitorMsgProcessor.IPCFilename = "/VisionSystemIPCDir/VSMonitorMsgSocketIPC";

        }
     

		/// <summary>
		/// Connects to vision system command socket.
		/// It is assumed the Connection Type, Connection Address 
		/// and Connection Port are already set.
		/// </summary>
		/// <returns>The to vision system command socket.</returns>
		public string ConnectToVisionSystemCommandResponseSocket()
		{
			return _cmdResponseProcess.ConnectToVisionSystemCmdResponseSocket();
		}

		public void DisconnectVisionSystemCommandResponseSocket()
		{
			_cmdResponseProcess.DisconnectVisionSystemCmdResponseSocket();
		}

		public VisionResponsePBMsg TransmitCommandResponseMessage(VisionCommandPBMsg cmd)
		{
			return _cmdResponseProcess.TransmitCommandResponseMessage(cmd);
		}

		/// <summary>
		/// Setup the socket for sending Vehicle Inertial State 
        /// and other Telemetry data.  This port is for sending information 
        /// to the Vision System in a Publish Subscribe format.  The Vision
        /// system subscribes to this information socket.
		/// </summary>
		public string ConnectPublishTelemeteryToVisionSystemSocket()
		{
			string errorMsg = null;
			try
			{
				_publishTelemetryToVisionSystemSocket = new PublisherSocket();
				string connect_str = string.Concat("tcp://", BridgeSystemTCPAddrString, ":", 
					                  BridgePublishTelemetryPort.ToString());
				_publishTelemetryToVisionSystemSocket.Bind(connect_str); 
			}
			catch(Exception ex)
			{
				errorMsg = string.Concat("ConnectVisionSystemPublishSocket Execption:", ex.Message);
			}
			return errorMsg;
		}

		public string PublishTelemetryMsgToVisionSystem(VSMessageWrapper msg)
		{
            string errorMsg = null;
            try
            {
                byte[] msgByteArray = msg.Serialize();
                _publishTelemetryToVisionSystemSocket.SendFrame(msgByteArray);
            }
            catch (Exception ex)
            {
                errorMsg = "PublishTelemetryToVisionSystem Exception: " + ex.Message;
            }
            return errorMsg;
		}

        public void DisconnectPublishTelemeterySocket()
        {
            if (_publishTelemetryToVisionSystemSocket != null)
            {
                _publishTelemetryToVisionSystemSocket.Close();
            }
        }

        /// <summary>
        /// Disconnect from the computer vision code.
        /// </summary>
        public void DisconnectAllSockets()
        {
			_cmdResponseProcess.DisconnectVisionSystemCmdResponseSocket();
            _vsResultsMsgProcessor.DisconnectSocket();
            _vsVideoStreamMsgProcessor.DisconnectSocket();
            _vsMonitorMsgProcessor.DisconnectSocket();
            DisconnectPublishTelemeterySocket();
        }
         

        #region IDisposable implementation

        public void Dispose()
        {
            DisconnectAllSockets();
        }

        #endregion
    }
}

