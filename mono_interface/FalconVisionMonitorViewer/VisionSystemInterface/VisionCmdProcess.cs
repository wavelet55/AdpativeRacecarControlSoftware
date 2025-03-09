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
using CarCANBusMonitor;

namespace FalconVisionMonitorViewer
{
	/// <summary>
	/// This class handles the Command/Response Messages between
	/// the Vision System Viewer and The Vision System
	/// </summary>
	public class VisionCmdProcess
	{
		private Bridge _visionBridge;

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
			_visionBridge = vsbridge;
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
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

        public string[] GetListOfManagerNames()
        {
            string[] mgrNames = null;
            ListOfManagerNamesPBMsg listOfMgrNamesPBMsg = null;
            VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "GetListOfManagerNames" };
            VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
			if (responseMsg.CmdResponseType == VisionResponsePBMsg.ResponseType_e.OK)
			{
				if (responseMsg.CmdResponseData != null)
				{
					try
					{
						listOfMgrNamesPBMsg = ListOfManagerNamesPBMsg.Deserialize(responseMsg.CmdResponseData);
                        if( listOfMgrNamesPBMsg.NumberOfManagers > 0 )
                            mgrNames = listOfMgrNamesPBMsg.ListOfManagerNames;
					} 
					catch (Exception ex)
					{
						listOfMgrNamesPBMsg = null;
					}
				}
			} 
            return mgrNames;
        }

		public ManagerStatsMsg GetManagerStats(string mgrName, out string cmdResponseMsg)
		{
			cmdResponseMsg = "Error";
			ManagerStatsMsg mgrStatsMsg = null;
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "GetManagerStats" };
			cmdMsg.CmdQualifier = mgrName;
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

		public string SendImageCaptureControlCmd(ImageCaptureControlMsg imageCaptureCtrlMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "ImageCaptureControlCmd" };
			cmdMsg.CmdData = imageCaptureCtrlMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

		public string SendStreamRecordControlCmd(StreamControlPBMsg srCtrlMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "StreamRecordImageControlMsg" };
			cmdMsg.CmdData = srCtrlMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

		public StreamControlPBMsg GetStreamRecordControlSettings(out string cmdResponseMsg)
		{
			cmdResponseMsg = "Error";
			StreamControlPBMsg srcMsg = null;
            try
            {
                VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg { Command = "GetStreamRecordImageControlMsg" };
                VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
                cmdResponseMsg = responseMsg.CmdResponseType.ToString();
                cmdResponseMsg += ":" + responseMsg.CmdResponseMessage;
                if (responseMsg.CmdResponseType == VisionResponsePBMsg.ResponseType_e.OK)
                {
                    if (responseMsg.CmdResponseData != null)
                    {
                        try
                        {
                            srcMsg = StreamControlPBMsg.Deserialize(responseMsg.CmdResponseData);
                        }
                        catch (Exception ex)
                        {
                            srcMsg = null;
                            cmdResponseMsg = string.Concat(cmdResponseMsg, "::ErrorDeserializing: ", ex.Message);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                srcMsg = null;
                cmdResponseMsg = string.Concat(cmdResponseMsg, "::GetStreamRecordControlSettings: Exception: ", ex.Message);
            }
			return srcMsg;
		}

		public string SendImageLoggingControlCmd(ImageLoggingControlMsg srCtrlMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "ImageLoggingControlMsg" };
			cmdMsg.CmdData = srCtrlMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

		public ImageLoggingControlMsg GetImageLoggingControlSettings(out string cmdResponseMsg)
		{
			cmdResponseMsg = "Error";
			ImageLoggingControlMsg srcMsg = null;
            try
            {
                VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg { Command = "GetImageLoggingControlMsg" };
                VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
                cmdResponseMsg = responseMsg.CmdResponseType.ToString();
                cmdResponseMsg += ":" + responseMsg.CmdResponseMessage;
                if (responseMsg.CmdResponseType == VisionResponsePBMsg.ResponseType_e.OK)
                {
                    if (responseMsg.CmdResponseData != null)
                    {
                        try
                        {
                            srcMsg = ImageLoggingControlMsg.Deserialize(responseMsg.CmdResponseData);
                        }
                        catch (Exception ex)
                        {
                            srcMsg = null;
                            cmdResponseMsg = string.Concat(cmdResponseMsg, "::ErrorDeserializing: ", ex.Message);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                srcMsg = null;
                cmdResponseMsg = string.Concat(cmdResponseMsg, "::GetImageLoggingControlSettings: Exception: ", ex.Message);
            }
			return srcMsg;
		}



		public string SendCameraCalControlMsg(CameraCalControlMsg cameraCalCtrlMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "CameraCalCommand" };
			cmdMsg.CmdData = cameraCalCtrlMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

		public string SendCameraParameersMsg(CameraParametersSetupPBMsg cameraParamslMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "CameraParametersSetupMessage" };
			cmdMsg.CmdData = cameraParamslMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

		public string SendFeatureMatchProcCtrlMsg(FeatureMatchProcCtrlPBMsg featureMatchCtrlMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "FeatureMatchProcCtrlMsg" };
			cmdMsg.CmdData = featureMatchCtrlMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

		public string SendVehicleInertialStatesMsgOnCmdPort(VehicleInterialStatesMsg visMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "VehicleInertialStatesMsg" };
			cmdMsg.CmdData = visMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

		public string SendCameraOrientationMsgOnCmdPort(CameraSteeringMsg coMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "CameraOrientationMsg" };
			cmdMsg.CmdData = coMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

		public string SendLinearActuatorPositionMsgOnCmdPort(LinearActuatorPositionCtrlPBMsg laCtrlMsg, LinearActuatorFunction_e laFnType)
		{
			string respStr = "";
            if(laFnType == LinearActuatorFunction_e.Brake || laFnType == LinearActuatorFunction_e.Accelerator)
            {
                string cmd = laFnType == LinearActuatorFunction_e.Brake ? "BrakePositionCtrlMsg" : "ThrottlePositionCtrlMsg";
			    VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = cmd };
			    cmdMsg.CmdData = laCtrlMsg.Serialize();
			    VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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
            }
			return respStr;
		}

        public string SendKarTeckActuatorConfigMsgOnCmdPort(KarTechLinearActuatorParamsPBMsg laCtrlMsg, LinearActuatorFunction_e laFnType)
		{
			string respStr = "";
            if(laFnType == LinearActuatorFunction_e.Brake || laFnType == LinearActuatorFunction_e.Accelerator)
            {
                string cmd = laFnType == LinearActuatorFunction_e.Brake ? "BrakeActuatorConfigMsg" : "ThrottleActuatorConfigMsg";
			    VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = cmd };
			    cmdMsg.CmdData = laCtrlMsg.Serialize();
			    VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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
            }
			return respStr;
		}

        public string SendKarTeckActuatorSetupMsgOnCmdPort(KarTechLinearActuatorSetupPBMsg laSetupMsg, LinearActuatorFunction_e laFnType)
		{
			string respStr = "";
            if(laFnType == LinearActuatorFunction_e.Brake || laFnType == LinearActuatorFunction_e.Accelerator)
            {
                string cmd = laFnType == LinearActuatorFunction_e.Brake ? "BrakeSetupMsg" : "ThrottleSetupMsg";
			    VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = cmd };
			    cmdMsg.CmdData = laSetupMsg.Serialize();
			    VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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
            }
			return respStr;
		}


		public KarTechLinearActuatorParamsPBMsg GetKarTeckActuatorConfigSettings(LinearActuatorFunction_e laFnType, out string cmdResponseMsg)
		{
			cmdResponseMsg = "Error";
			KarTechLinearActuatorParamsPBMsg srcMsg = null;
            try
            {
                string cmd = laFnType == LinearActuatorFunction_e.Brake ? "GetBrakeActuatorConfigParams" : "GetThrottleActuatorConfigParams";
                VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg { Command = cmd };
                VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
                cmdResponseMsg = responseMsg.CmdResponseType.ToString();
                cmdResponseMsg += ":" + responseMsg.CmdResponseMessage;
                if (responseMsg.CmdResponseType == VisionResponsePBMsg.ResponseType_e.OK)
                {
                    if (responseMsg.CmdResponseData != null)
                    {
                        try
                        {
                            srcMsg = KarTechLinearActuatorParamsPBMsg.Deserialize(responseMsg.CmdResponseData);
                        }
                        catch (Exception ex)
                        {
                            srcMsg = null;
                            cmdResponseMsg = string.Concat(cmdResponseMsg, "::ErrorDeserializing: ", ex.Message);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                srcMsg = null;
                cmdResponseMsg = string.Concat(cmdResponseMsg, "::GetKarTeckActuatorConfigSettings: Exception: ", ex.Message);
            }
			return srcMsg;
		}

        public string SendSteeringTorqueContronMsgOnCmdPort(SteeringTorqueCtrlPBMsg steeringCtrlMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "SteeringTorqueControlMsg" };
			cmdMsg.CmdData = steeringCtrlMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

        public string SendIMUControlMsgOnCmdPort(IMUCommandResponsePBMsg imuCtrlMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "IMUCommandMsg" };
			cmdMsg.CmdData = imuCtrlMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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


		public bool SendVehicleInertialStatesMsgOnTelemetryPort(VehicleInterialStatesMsg visMsg)
		{
            bool error = true;
            string errorMsg = null;

            VisionBridge.Messages.VSMessageWrapper vsMsgWrapper = new VSMessageWrapper();
            vsMsgWrapper.MsgName = "VehicleInertialStates";
            vsMsgWrapper.MsgData = visMsg.Serialize();
            vsMsgWrapper.MsgDataSize = vsMsgWrapper.MsgData.Length;

            errorMsg = _visionBridge.PublishTelemetryMsgToVisionSystem(vsMsgWrapper);
            if (errorMsg != null)
            {
                error = true;
            }
            else
            {
                error = false;
            }
            return error;
		}

		public bool SendCameraOrientationMsgOnTelemetryPort(CameraSteeringMsg coMsg)
		{
            bool error = true;
            string errorMsg = null;

            VisionBridge.Messages.VSMessageWrapper vsMsgWrapper = new VSMessageWrapper();
            vsMsgWrapper.MsgName = "CameraOrientation";
            vsMsgWrapper.MsgData = coMsg.Serialize();
            vsMsgWrapper.MsgDataSize = vsMsgWrapper.MsgData.Length;

            errorMsg = _visionBridge.PublishTelemetryMsgToVisionSystem(vsMsgWrapper);
            if (errorMsg != null)
            {
                error = true;
            }
            else
            {
                error = false;
            }
            return error;
		}



		public ImageCaptureStatusMsg GetImageCaptureStatus(out string cmdResponseMsg)
		{
			cmdResponseMsg = "Error";
			ImageCaptureStatusMsg statsMsg = null;
            try
            {
                VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg { Command = "GetImageCaptureStatus" };
                VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
                cmdResponseMsg = responseMsg.CmdResponseType.ToString();
                cmdResponseMsg += ":" + responseMsg.CmdResponseMessage;
                if (responseMsg.CmdResponseType == VisionResponsePBMsg.ResponseType_e.OK)
                {
                    if (responseMsg.CmdResponseData != null)
                    {
                        try
                        {
                            statsMsg = ImageCaptureStatusMsg.Deserialize(responseMsg.CmdResponseData);
                        }
                        catch (Exception ex)
                        {
                            statsMsg = null;
                            cmdResponseMsg = string.Concat(cmdResponseMsg, "::ErrorDeserializing: ", ex.Message);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                statsMsg = null;
                cmdResponseMsg = string.Concat(cmdResponseMsg, "::GetImageCaptureStatus: Exception: ", ex.Message);
            }
			return statsMsg;
		}

		public ImageCaptureControlMsg GetImageCaptureControlSettings(out string cmdResponseMsg)
		{
			cmdResponseMsg = "Error";
			ImageCaptureControlMsg statsMsg = null;
            try
            {
                VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg { Command = "GetImageCaptureControlSettings" };
                VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
                cmdResponseMsg = responseMsg.CmdResponseType.ToString();
                cmdResponseMsg += ":" + responseMsg.CmdResponseMessage;
                if (responseMsg.CmdResponseType == VisionResponsePBMsg.ResponseType_e.OK)
                {
                    if (responseMsg.CmdResponseData != null)
                    {
                        try
                        {
                            statsMsg = ImageCaptureControlMsg.Deserialize(responseMsg.CmdResponseData);
                        }
                        catch (Exception ex)
                        {
                            statsMsg = null;
                            cmdResponseMsg = string.Concat(cmdResponseMsg, "::ErrorDeserializing: ", ex.Message);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                statsMsg = null;
                cmdResponseMsg = string.Concat(cmdResponseMsg, "::GetImageCaptureControlSettings: Exception: ", ex.Message);
            }
			return statsMsg;
		}



		public string SendVisionProcessControlCmd(VisionProcessingControlMsg visionProcessCtrlMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "VisionProcessingControlCmd" };
			cmdMsg.CmdData = visionProcessCtrlMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

		public VisionProcessingControlMsg GetVisionProcessControlStatus(out string cmdResponseMsg)
		{
			cmdResponseMsg = "Error";
			VisionProcessingControlMsg statsMsg = null;
            try
            {
                VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg { Command = "VisionProcessCmdStatus" };
                VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
                cmdResponseMsg = responseMsg.CmdResponseType.ToString();
                cmdResponseMsg += ":" + responseMsg.CmdResponseMessage;
                if (responseMsg.CmdResponseType == VisionResponsePBMsg.ResponseType_e.OK)
                {
                    if (responseMsg.CmdResponseData != null)
                    {
                        try
                        {
                            statsMsg = VisionProcessingControlMsg.Deserialize(responseMsg.CmdResponseData);
                        }
                        catch (Exception ex)
                        {
                            statsMsg = null;
                            cmdResponseMsg = string.Concat(cmdResponseMsg, "::ErrorDeserializing: ", ex.Message);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                statsMsg = null;
                cmdResponseMsg = string.Concat(cmdResponseMsg, "::GetVisionProcessControlStatus: Exception: ", ex.Message);
            }
			return statsMsg;
		}

		public string SetupGeoCoordCmd(double centerLatDeg, double centerLonDeg, double gndAltMSL, int convType)
		{
			string respStr = "";
			GeoCoordinateSystemSetupMsg geoCSetupMsg = new GeoCoordinateSystemSetupMsg();
            geoCSetupMsg.CenterLatitudeDegrees = centerLatDeg;
            geoCSetupMsg.CenterLongitudeDegrees = centerLonDeg;
            geoCSetupMsg.GroundAltitudeMSL = gndAltMSL;
            geoCSetupMsg.GeoCoordinateSystemConversionType = GeoCoordinateSystemSetupMsg.GeoCoordinateSystemConversionType_e.Linear;
            if (convType == 1)
                geoCSetupMsg.GeoCoordinateSystemConversionType = GeoCoordinateSystemSetupMsg.GeoCoordinateSystemConversionType_e.WGS84_Relative;
            else if (convType == 2)
                geoCSetupMsg.GeoCoordinateSystemConversionType = GeoCoordinateSystemSetupMsg.GeoCoordinateSystemConversionType_e.WGS84_Map;

			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "SetupGeoCoordinateSystem" };
			cmdMsg.CmdData = geoCSetupMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

        public GeoCoordinateSystemSetupMsg GetGeoCoordSetupCmd()
        {
            GeoCoordinateSystemSetupMsg geoCSetupMsg = null;
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "GetGeoCoordinateSetup" };
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
			if (responseMsg.CmdResponseType == VisionResponsePBMsg.ResponseType_e.OK)
			{
				if (responseMsg.CmdResponseData != null)
				{
					try
					{
						geoCSetupMsg = GeoCoordinateSystemSetupMsg.Deserialize(responseMsg.CmdResponseData);
					} 
					catch (Exception ex)
					{
						geoCSetupMsg = null;
					}
				}
			} 
			return geoCSetupMsg;
        }

        public LatLonXYConversionMsg GeoCoordLatLonXYConvCmd(bool latLonToXY, double p1, double p2)
        {
            VisionCommandPBMsg cmdMsg;
            LatLonXYConversionMsg latLonXYConvMsg = new LatLonXYConversionMsg();
            if (latLonToXY)
            {
                latLonXYConvMsg.LatitudeDegrees = p1;
                latLonXYConvMsg.LongitudeDegrees = p2;
                latLonXYConvMsg.LatLonToXYConversion = true;
            }
            else  //XY to Lat/Lon
            {
                latLonXYConvMsg.X_PosMeters = p1;
                latLonXYConvMsg.Y_PosMeters = p2;
                latLonXYConvMsg.LatLonToXYConversion = false;
            }

            cmdMsg = new VisionCommandPBMsg{ Command = "LatLonXYConversion" };
            cmdMsg.CmdData = latLonXYConvMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
			if (responseMsg.CmdResponseType == VisionResponsePBMsg.ResponseType_e.OK)
			{
				if (responseMsg.CmdResponseData != null)
				{
					try
					{
						latLonXYConvMsg = LatLonXYConversionMsg.Deserialize(responseMsg.CmdResponseData);
					} 
					catch (Exception ex)
					{
						latLonXYConvMsg = null;
					}
				}
			} 
			return latLonXYConvMsg;
        }

        public string SendHeadTrackingControlMsg(HeadTrackingControlPBMsg ctrlMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "HeadTrackingControlMsg" };
			cmdMsg.CmdData = ctrlMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

        public string SendVidereSystemControlMsg(VidereSystemControlPBMsg ctrlMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "VidereSystemControlMsg" };
			cmdMsg.CmdData = ctrlMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

        public string SendHeadOrientationCommandMsg(HeadOrientationControlPBMsg ctrlMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "HeadOrientationControlMsg" };
			cmdMsg.CmdData = ctrlMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

        public string SendVehicleControlParametersMsg(VehicleControlParametersPBMsg ctrlMsg)
		{
			string respStr = "";
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "SetVehicleControlParametersMsg" };
			cmdMsg.CmdData = ctrlMsg.Serialize();
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
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

        public VehicleControlParametersPBMsg GetVehicleControlParameters()
        {
            VehicleControlParametersPBMsg vcpMsg = null;
			VisionCommandPBMsg cmdMsg = new VisionCommandPBMsg{ Command = "GetVehicleControlParametersMsg" };
			VisionResponsePBMsg responseMsg = _visionBridge.TransmitCommandResponseMessage(cmdMsg);
			if (responseMsg.CmdResponseType == VisionResponsePBMsg.ResponseType_e.OK)
			{
				if (responseMsg.CmdResponseData != null)
				{
					try
					{
						vcpMsg = VehicleControlParametersPBMsg.Deserialize(responseMsg.CmdResponseData);
					} 
					catch (Exception ex)
					{
						vcpMsg = null;
					}
				}
			} 
			return vcpMsg;
        }

	}
}

