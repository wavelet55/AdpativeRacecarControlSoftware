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
    /// BridgeCommandResponseProcess
    /// Connects to the Vision System Command-Response Socket
    /// and provides the lower-level processing of Command-Response
    /// messages.  For each command message send to the Vision System,
    /// the Vision System will respond with a response message.
    /// This is the primary control message path to the Vision System.
    /// </summary>
	public class BridgeCommandResponseProcess
	{
		public BridgeConnectionType_e BridgeConnectionType = BridgeConnectionType_e.tcp;

		private string _TCPAddrString = "localhost";
		/// <summary>
		/// A TCP IP Address in the format:
		/// "127.0.0.1" or "localhost
		/// </summary>
		/// <value>IP address string.</value>
		public string TCPAddrString 
		{
			get { return _TCPAddrString; }
			set { _TCPAddrString = string.IsNullOrEmpty(value) ? "localhost" : value.Trim().ToLower(); }
		}

		private int _TCPPort = 5555;
		/// <summary>
		/// Socket Port for tcp connections.
		/// </summary>
		/// <value>TCP Port Number</value>
		public int TCPPort
		{
			get { return _TCPPort; }
			set { _TCPPort = value < 1 ? 1 : value > 65535 ? 65535 : value; }
		}

		private string _IPCFilename = "/IPCComm/VSRequestIPC";
		/// <summary>
		/// Inter-Process Comm Filename
		/// </summary>
		/// <value>The vision system command port.</value>
		public string IPCFilename 
		{
			get { return _IPCFilename; }
			set { _IPCFilename = string.IsNullOrEmpty(value) ? "/IPCComm/VSRequestIPC" : value.Trim(); }
		}

		private RequestSocket _cmdResponseSocket = null;
		/// <summary>
		/// Is the Vision System Request Process Socket Connected.
		/// </summary>
		/// <value><c>true</c> if this instance is vision command socket connected; otherwise, <c>false</c>.</value>
		public bool IsConnected 
		{ 
			get { return _cmdResponseSocket != null ? true : false; }
		}

		//private readonly NetMQContext _zmqContext;

		//private NetMQMessage _transmitZMQMsg; 
		//private NetMQMessage _receiveZMQMsg; 
		private byte[] _receiveZMQMsgByteArray = null;

		public BridgeCommandResponseProcess()
		{
			//_zmqContext = zmqContext;
			//_transmitZMQMsg = new NetMQMessage(2);
			//_receiveZMQMsg = new NetMQMessage(2);
		}

		private double _maxResponseWaitTimeSeconds = 1.0;
		/// <summary>
		/// The Maximum time to wait for the Vision System to respond to 
		/// a Request/Command.  If the vision system fails to respond in this
		/// timeframe, and error will be declared and the socket will be reset.
		/// </summary>
		/// <value>The max response wait time seconds.</value>
		public double MaxResponseWaitTimeSeconds 
		{
			get { return _maxResponseWaitTimeSeconds; }
			set { _maxResponseWaitTimeSeconds = value < 0.01 ? 0.01 : value; }
		}


		/// <summary>
		/// Connects to vision system request socket.
		/// </summary>
		/// <returns>The to vision system command socket.</returns>
		/// <param name="connType">Conn type.</param>
		/// <param name="connAddr">Conn address: TCP Addr or IPC Filename</param>
		/// <param name="connPortNumber">Conn port number.</param>
		public string ConnectToVisionSystemCmdResponseSocket(BridgeConnectionType_e connType, 
                                                            string connAddr, int connPortNumber)
		{
			BridgeConnectionType = connType;
			if( connType == BridgeConnectionType_e.ipc )
				IPCFilename = connAddr;
			else
				TCPAddrString = connAddr;
			
			TCPPort = connPortNumber;
			return ConnectToVisionSystemCmdResponseSocket();
		}

		/// <summary>
		/// Connects to vision system command socket.
		/// It is assumed the Connection Type, Connection Address 
		/// and Connection Port are already set.
		/// </summary>
		/// <returns>The to vision system command socket.</returns>
		public string ConnectToVisionSystemCmdResponseSocket()
		{
			string errorMsg = null;
			if(_cmdResponseSocket == null)
			{
				if (errorMsg == null)
				{
					try
					{
						string connStr;
						if( BridgeConnectionType == BridgeConnectionType_e.ipc )
						{
							connStr = string.Concat("ipc://", IPCFilename);
						}
						else
						{
							connStr = string.Concat("tcp://", TCPAddrString, ":", TCPPort.ToString());
						}

						_cmdResponseSocket = new RequestSocket();

						if (_cmdResponseSocket != null)
						{
							//_cmdResponseSocket.Options.Endian = NetMQ.Endianness.Little;
							_cmdResponseSocket.Connect(connStr);
							_cmdResponseSocket.ReceiveReady += new EventHandler<NetMQSocketEventArgs>(EventClientReceiveMsg);
						} 
						else
						{
							errorMsg = "CreateRequestSocket Failed";
						}
					} 
					catch (Exception ex)
					{
						errorMsg = string.Concat("ConnectToVisionSystemRequestSocket Execption:", ex.Message);
					}
				}
			}
			return errorMsg;
		}

		public void DisconnectVisionSystemCmdResponseSocket()
		{
			if (_cmdResponseSocket != null)
			{
				_cmdResponseSocket.Close();
				_cmdResponseSocket = null;
			}
		}

		/// <summary>
		/// Reset the Request Request Client Socket.
		/// This will be used if there are issues on the socket
		/// or if the Server does not respond in a timely fashion.
		/// </summary>
		public void ResetVisionSystemCmdRequestSocket()
		{
			DisconnectVisionSystemCmdResponseSocket();
			ConnectToVisionSystemCmdResponseSocket();
		}

		/// <summary>
		/// Convert a string to an array of ASCII bytes.
		/// </summary>
		/// <param name="strVal"></param>
		/// <returns></returns>
		public byte[] StringToByteArray(string strVal)
		{
			return Encoding.ASCII.GetBytes(strVal);
		}

		/// <summary>
		/// Convert a ASCII byte array to a string.
		/// This method is string encoding agnostic.
		/// </summary>
		/// <param name="byteArray"></param>
		/// <returns></returns>
		public string ByteArrayToString(byte[] byteArray)
		{
			return ASCIIEncoding.ASCII.GetString(byteArray);
		}

		/// <summary>
		/// Event Driven Receive on the Client Command/Response Socket
		/// </summary>
		/// <param name="reqSocket"></param>
		/// <param name="netMQArgs"></param>
		private void EventClientReceiveMsg(object socket, NetMQSocketEventArgs netMQArgs)
		{
			_receiveZMQMsgByteArray = null;
			if (netMQArgs.IsReadyToReceive && socket != null && socket is RequestSocket)
			{
				RequestSocket reqSocket = (RequestSocket)socket;
				_receiveZMQMsgByteArray = reqSocket.ReceiveFrameBytes();

				//reqSocket.TryReceiveFrameBytes(out _receiveZMQMsgByteArray);
				//If response message is a byte array use this approach:
				//byte[] responseByteArray;
				//_cmdReqClientSocket.TryReceiveFrameBytes(out responseByteArray);
				//rxMsg = ByteArrayToString(responseByteArray);
			}
		}

		//ToDo:  change this to sending and receiving generic message objects.
		public VisionResponsePBMsg TransmitCommandResponseMessage(VisionCommandPBMsg cmd)
		{
			VisionResponsePBMsg responseMessage = null;
			_receiveZMQMsgByteArray = null;   //Reset the Reply Message
			TimeSpan timeSpan = new TimeSpan(0, 0, 0, (int)MaxResponseWaitTimeSeconds, (int)((MaxResponseWaitTimeSeconds % 1.0) * 1000.0));
			bool resetSocket = true;

			if (_cmdResponseSocket != null)
			{
				try
				{
					//_transmitZMQMsg.Clear();
					//_transmitZMQMsg.Append(cmd.command)
					byte[] cmdByteArray = cmd.Serialize();

					//NetMQMessage mqMsg = new NetMQMessage(1);
					//mqMsg.Append(cmdType);
					//mqMsg.Append(cmdByteArray);

					//Send the Message to the Server Side
					//if (_requestSocket.TrySendMultipartMessage(mqMsg))
					//if (_requestSocket.TrySendFrame(cmdByteArray))
					_cmdResponseSocket.SendFrame(cmdByteArray);
					{
						//Now wait for the server to respond... the server must
						//respond to each command message.
						//We only want to wait a max amount of time incase the server is down
						//for some reason... otherwize we will lock up here forever.
						//The Receive message is actually recieved by the EventClientReceiveMsg(...)
						//event.  The Poll mechanizm return as soon as a message is received, or 
						//after the timeout.
						if (_cmdResponseSocket.Poll(timeSpan))
						{
							if (_receiveZMQMsgByteArray != null)
							{
								responseMessage = VisionResponsePBMsg.Deserialize(_receiveZMQMsgByteArray);
								resetSocket = false;
							} else
							{
								responseMessage = new VisionResponsePBMsg();
								responseMessage.CmdResponseType = VisionResponsePBMsg.ResponseType_e.ERROR;
								responseMessage.CmdResponseMessage = "Vision System did not receive response message.";
								resetSocket = true;
							}
						} else 
						{
							responseMessage = new VisionResponsePBMsg();
							responseMessage.CmdResponseType = VisionResponsePBMsg.ResponseType_e.ERROR;
							responseMessage.CmdResponseMessage = "Vision System did not respond to the command.";
							resetSocket = true;
						}
					}
				} catch (Exception ex)
				{
					responseMessage = new VisionResponsePBMsg();
					responseMessage.CmdResponseType = VisionResponsePBMsg.ResponseType_e.ERROR;
					responseMessage.CmdResponseMessage = "Request Socket Exception: " + ex.Message;
					resetSocket = true;
				}
				if (resetSocket)
				{
					ResetVisionSystemCmdRequestSocket();
				}
			} 
			else
			{
				responseMessage = new VisionResponsePBMsg();
				responseMessage.CmdResponseType = VisionResponsePBMsg.ResponseType_e.ERROR;
				responseMessage.CmdResponseMessage = "Vision System Request Socket Not Connected.";
			}
			return responseMessage;
		}



	}
}

