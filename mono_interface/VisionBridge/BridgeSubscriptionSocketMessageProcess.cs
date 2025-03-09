/* ****************************************************************
 * Vision Bridge
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
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
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using VisionBridge.Messages;

namespace VisionBridge
{
    /// <summary>
    /// BridgeSubscriptionSocketMessageProcess
    /// The Vision System has a number of Publish Subscribe Sockets
    /// for publishing results of the Vision System.  The Vision 
    /// System publishes various result, images, and performance
    /// messages on the different sockets.  This is a generic
    /// Message processor for for a single socket.  This Class
    /// is designed to connect to a Vision Socket as a Subscriber
    /// and then handle all the low level processing of the messages
    /// coming in over the socket from the Vision System.
    /// The user provides Handlers for each of the messages coming in.
    /// The processor uses Async processess so each mesage is handled 
    /// as it arrives.
    /// </summary>
    public class BridgeSubscriptionSocketMessageProcess
    {

		public BridgeConnectionType_e SocketConnectionType = BridgeConnectionType_e.tcp;

		private string _TCPAddrString = "localhost";
		/// <summary>
		/// A TCP IP Address in the format:
		/// "127.0.0.1" or "localhost
        /// of the Vision System... the Vision System may be on
        /// a different computer than the Bridge Host.
		/// </summary>
		/// <value>IP address string.</value>
		public string VisionSystemTCPAddrString 
		{
			get { return _TCPAddrString; }
			set 
            { 
                _TCPAddrString = string.IsNullOrEmpty(value) ? "localhost" : value.Trim().ToLower();             
            }
		}

		private int _TCPPort = 5556;
		/// <summary>
		/// Socket Port for tcp connections.
		/// </summary>
		/// <value>TCP Port Number</value>
		public int TCPPort
		{
			get { return _TCPPort; }
			set { _TCPPort = value < 1 ? 1 : value > 65535 ? 65535 : value; }
		}

		private string _IPCFilename = "/VisionSystemIPCDir/VSSubscriptionFNIPC";
		/// <summary>
		/// Inter-Process Comm Filename
        /// This can be used on Linux computers if the bridge host is on the 
        /// same machine as the Vision System.
        /// IPC is not current supported on Windows Systems by ZMQ.
		/// </summary>
		/// <value>The vision system command port.</value>
		public string IPCFilename 
		{
			get { return _IPCFilename; }
			set { _IPCFilename = string.IsNullOrEmpty(value) ? "/VisionSystemIPCDir/VSRequestIPC" : value.Trim(); }
		}

        private int _socketHighWaterMark = 25;
        /// <summary>
        /// The Socket High Water Mark is the maximum number of messages that will
        /// be allowed to reside in the message queue within the socket.
        /// </summary>
        public int SocketHighWaterMark
        {
            get { return _socketHighWaterMark; }
            set { _socketHighWaterMark = value < 2 ? 2 : value > 1000 ? 1000 : value; }
        }

        private SubscriberSocket _subscriberVSMsgSocket = null;

		/// <summary>
		/// Is the Vision System Subcriber Socket Connected.  
		/// </summary>
		/// <value><c>true</c> if this instance is vision message socket connected; otherwise, <c>false</c>.</value>
		public bool IsVSSocketConnected 
		{ 
			get { return _subscriberVSMsgSocket != null ? true : false; }
		}

        private UInt32 _processVisionMessageErrorCount = 0;
        /// <summary>
        /// A counter that can be checked to see if there are errors
        /// associated with the front-end processing of the Vision Results Messages.</summary>
        public UInt32 ProcessVisionMessageErrorCount
        {
            get { return _processVisionMessageErrorCount; }
        }

        private string _processVisionErrorMessage = "";
        /// <summary>
        /// The last error message associated with the Message Processing.
        /// </summary>
        public string ProcessVisionErrorMessage
        {
            get { return _processVisionErrorMessage; }
        }

        private UInt32 _processVisionMessageWarningCount = 0;
        /// <summary>
        /// A counter that can be checked to see if there are errors
        /// associated with the front-end processing of the Vision Results Messages.</summary>
        public UInt32 ProcessVisionMessageWarningCount
        {
            get { return _processVisionMessageWarningCount; }
        }

        private string _processVisionWarningMessage = "";
        /// <summary>
        /// The last error message associated with the Message Processing.
        /// </summary>
        public string ProcessVisionWarningMessage
        {
            get { return _processVisionWarningMessage; }
        }


        //Dictionary of Message Handlers.
        Dictionary<string, Action<string, byte[]>> _msgHandlerDictionary; 


		//private readonly NetMQContext _zmqContext;

		private byte[] _receiveZMQMsgByteArray = null;

        // For canceling out of the async/await routines.
        private CancellationTokenSource _cts;

        private bool _stopProcessingMessages = false;

        public BridgeSubscriptionSocketMessageProcess()
        {
            _cts = new CancellationTokenSource();
            _stopProcessingMessages = false;
            _msgHandlerDictionary = new Dictionary<string, Action<string, byte[]>>();
        }

        public void ClearErrors()
        {
            _processVisionMessageErrorCount = 0;
            _processVisionErrorMessage = "";
        }

        public void ClearWarnings()
        {
            _processVisionMessageWarningCount = 0;
            _processVisionWarningMessage = "";
        }


        /// <summary>
        /// Add a Message Handler for each Incoming Message to be processed.
        /// The msgHandlerName must match the MessageName that is contained 
        /// in the received VSMessageWrapper.MsgName exactly (case sensitive).
        /// Each message handler must handle the message to be processed including 
        /// all errors.  The message handler should no throw any exceptions at this
        /// could kill the handling of any further messages from this Vision System 
        /// socket.  
        /// The Action method is passed a string which may contain parameters for the
        /// message handler.  The byte array contains the message payload and can be
        /// most anything.  The message handler is responsible for interpreting the 
        /// message data and processing the message data.
        /// </summary>
        /// <param name="msgHandlerName"></param>
        /// <param name="handler"></param>
        /// <returns></returns>
        public void AddMessageHandler(string msgHandlerName, Action<string, byte[]> handler)
        {
            _msgHandlerDictionary[msgHandlerName] = handler;
        }

        /// <summary>
        /// Connect to the Vision System Message Socket
        /// The Vision Processing publishes messages on this socket.
        /// All messages are wrapped in the VSMessageWrapper message.
        /// Check the Error Count and Error message after the connection
        /// attempt to see if there were errors connecting to the socket.
        /// </summary>
        public async Task ConnectToVisionSubscriptionSocketAsync()
        {
			string connStr;
			if( SocketConnectionType == BridgeConnectionType_e.ipc )
			{
				connStr = string.Concat("ipc://", IPCFilename);
			}
			else
			{
				connStr = string.Concat("tcp://", VisionSystemTCPAddrString, ":", TCPPort.ToString());
			}

            _stopProcessingMessages = false;
			using (_subscriberVSMsgSocket = new SubscriberSocket())
            {
                if (_subscriberVSMsgSocket != null)
                {
                    try
                    {
                        // Connect the client to the server
                        _subscriberVSMsgSocket.Options.ReceiveHighWatermark = _socketHighWaterMark;
                        _subscriberVSMsgSocket.Connect(connStr);
                        _subscriberVSMsgSocket.SubscribeToAnyTopic();
                    }
                    catch (Exception ex)
                    {
                         _processVisionErrorMessage = "ConnectToVisionSubscriptionSocketAsync: excption connecting to the Subscriber Socket! ";
                         _processVisionErrorMessage += "Exception: " + ex.Message;
                         ++_processVisionMessageErrorCount;
                         _subscriberVSMsgSocket = null;
                         return;
                    }
                }
                else
                {
                    _processVisionErrorMessage = "ConnectToVisionSubscriptionSocketAsync: could not obtain a Subscriber Socket!";
                    ++_processVisionMessageErrorCount;
                    return;
                }

				await ProcessVisionSystemMessage(_subscriberVSMsgSocket);
            }
        }

        public void DisconnectSocket()
        {
            _stopProcessingMessages = true;
            if (_cts != null)
            {
                _cts.Cancel();
            }
            if (_subscriberVSMsgSocket != null)
            {
                _subscriberVSMsgSocket.Close();
                _subscriberVSMsgSocket = null;
            }
        }


        private async Task ProcessVisionSystemMessage(SubscriberSocket subSocket)
        {
            if (!_stopProcessingMessages && subSocket != null)
            {
                byte[] rxData = await Task.Run<byte[]>(() => subSocket.ReceiveFrameBytes(), _cts.Token);
                if (rxData != null)
                {
                    try
                    {
                        VSMessageWrapper rxMsg = VSMessageWrapper.Deserialize(rxData);
                        if (_msgHandlerDictionary.ContainsKey(rxMsg.MsgName))
                        {
                            //Process the Message with the Associated Message Handler.
                            Action<string, byte[]> handler = _msgHandlerDictionary[rxMsg.MsgName];
                            handler(rxMsg.MsgQualifier, rxMsg.MsgData);
                        }
                        else
                        {
                            ++_processVisionMessageWarningCount;
                            _processVisionWarningMessage = rxMsg.MsgName + ": Not Handled";
                        }
                    }
                    catch (Exception ex)
                    {
                        ++_processVisionMessageErrorCount;
                        _processVisionErrorMessage = "ProcessVisionSystemMessage Exception: " + ex.Message;
                    }
                    finally
                    {
                        //Release the byte array so it can be reclaimed.
                        rxData = null;
                    }
                }
            }
            if (!_stopProcessingMessages)
            {
                await ProcessVisionSystemMessage(subSocket);
            }
        }

        public void Dispose()
        {
            DisconnectSocket();
        }

    }
}
