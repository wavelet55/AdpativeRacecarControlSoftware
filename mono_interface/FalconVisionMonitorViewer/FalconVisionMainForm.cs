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
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using VisionBridge;
using VisionBridge.Messages;
using GeoCoordinateSystemNS;
using CarCANBusMonitor;
using CarCANBusMonitor.Widgets;

namespace FalconVisionMonitorViewer
{
    public partial class FalconVisionViewerMainForm : Form
    {

        private Bridge _visionBridge;
        private long _numMessages;
        private bool _connectedToVisionBridge = false;

        private VisionCmdProcess _visionCmdProcess;

        private bool _sendTelemetryToVisionSystem = false;
        private bool _imageCaptureEnabled = false;
        private bool _targetProcssingEnabled = false;
        private bool _videoStreamEnabled = false;
        private bool _recordingEnabled = false;

        private int MgrStatsCounter = 0;

        private ImageCaptureStatusMsg _imageCaptureStatusMsg;

        private VisionProcessingControlMsg _visionProcessCmdStatusMsg;

        private GeoCoordinateSystem _geoCoordinateSystem;

        private DceEPAS _steeringControlObj;
        private KarTeckLinearActuator _throttleControlObj;
        private KarTeckLinearActuator _brakeControlObj;

        public FalconVisionViewerMainForm()
        {
            InitializeComponent();

            _numMessages = 0;
            _visionBridge = new Bridge();
            _visionCmdProcess = new VisionCmdProcess(_visionBridge);

            _geoCoordinateSystem = new GeoCoordinateSystem();

            _steeringControlObj = new DceEPAS(_visionCmdProcess);
            _throttleControlObj = new KarTeckLinearActuator(LinearActuatorFunction_e.Accelerator, _visionCmdProcess);
            _brakeControlObj = new KarTeckLinearActuator(LinearActuatorFunction_e.Brake, _visionCmdProcess);

            steeringControlWidget1.EPAS_Obj = _steeringControlObj;
            steeringControlWidget1.BackColor = System.Drawing.Color.Aqua;

            ThrottlePositionControlWidget.Actuator = _throttleControlObj;
            ThrottlePositionControlWidget.SetFunctionName("Throttle");
            ThrottlePositionControlWidget.BackColor = System.Drawing.Color.LightGreen;

            throttleLAPosCtrlSetupTab.Actuator = _throttleControlObj;
            throttleLAPosCtrlSetupTab.SetFunctionName("Throttle");
            throttleLAPosCtrlSetupTab.BackColor = System.Drawing.Color.LightGreen;

            throttleLAConfigSetupTab.LinearActuator = _throttleControlObj;
            throttleLAConfigSetupTab.UpdateParameters();
            throttleLAConfigSetupTab.BackColor = System.Drawing.Color.LightGreen;

            BrakePositionControlWidget.Actuator = _brakeControlObj;
            BrakePositionControlWidget.SetFunctionName("Brake");
            BrakePositionControlWidget.BackColor = System.Drawing.Color.LightSalmon;

            brakeLAPosCtrlSetupTab.Actuator = _brakeControlObj;
            brakeLAPosCtrlSetupTab.SetFunctionName("Brake");
            brakeLAPosCtrlSetupTab.BackColor = System.Drawing.Color.LightSalmon;

            brakeLAConfigSetupTab.LinearActuator = _brakeControlObj;
            brakeLAConfigSetupTab.UpdateParameters();
            brakeLAConfigSetupTab.BackColor = System.Drawing.Color.LightSalmon;

            gpsFixWidget1.BackColor = System.Drawing.Color.YellowGreen;

            tbVSIPAddr.Text = "192.168.1.78";
            tbHostIPAddr.Text = "192.168.1.105";

            timerMgrStatsUpdate.Enabled = false;

            videreSystemStateControlForm.VisionCmdProc = _visionCmdProcess;
            videreSystemStateControlForm.ResetAllVehicleActuators = ResetAllVehicleActuators;
            videreSystemStateControlForm.BackColor = System.Drawing.Color.LightGreen;

            imuControlFrm.VisionCmdProc = _visionCmdProcess;

            imageCaptureControl1.VisionCmdProc = _visionCmdProcess;
            imageCaptureSetupAndStatus1.VisionCmdProc = _visionCmdProcess;

            imageProcessControl1.VisionCmdProc = _visionCmdProcess;

            blobDetectorParameters1.VisionCmdProc = _visionCmdProcess;

            vehicleControlParametersWidget1.VisionCmdProc = _visionCmdProcess;

            //Setup Manager Stats Controls
            MgrStatsCtrl_MgrNo_1.VisionCmdProc = _visionCmdProcess;
            MgrStatsCtrl_MgrNo_2.VisionCmdProc = _visionCmdProcess;
            MgrStatsCtrl_MgrNo_3.VisionCmdProc = _visionCmdProcess;
            MgrStatsCtrl_MgrNo_4.VisionCmdProc = _visionCmdProcess;

            MgrStatsCtrl_MgrNo_1.MgrIndexNo = 0;
            MgrStatsCtrl_MgrNo_2.MgrIndexNo = 1;
            MgrStatsCtrl_MgrNo_3.MgrIndexNo = 2;
            MgrStatsCtrl_MgrNo_4.MgrIndexNo = 3;

            cameraOrientationControl1.VisionCmdProc = _visionCmdProcess;

            cameraParametersSetupWidget1.VisionCmdProc = _visionCmdProcess;

            streamRecordControlWidget1.VisionCmdProc = _visionCmdProcess;

            cameraCalControl1.VisionCmdProc = _visionCmdProcess;
            cameraCalControl1.CameraCalChessBdInp = cameraCalChessBdInput1;
            cameraCalControl1.CameraMountCorrInp = cameraMountCorrectionInput1;

            headOrientationCalWidget1.VisionCmdProc = _visionCmdProcess;

            headOrientationControlWidget1.VisionCmdProc = _visionCmdProcess;

            featureMatchProcessControl1.VisionCmdProc = _visionCmdProcess;

            processTimerStatus1.VisionCmdProc = _visionCmdProcess;

            geoCoordinateSystemSetup1.VisionCmdProc = _visionCmdProcess;
            geoCoordinateSystemSetup1.GeoCoordinateSys = _geoCoordinateSystem;

            UavInertialStatesFromImageInfo.SetDisplayOrSendFormType(true);
            UavInertialStatesFromImageInfo.GeoCoordinateSys = _geoCoordinateSystem;

            SendInertialStatesToUAVCtrl.SetDisplayOrSendFormType(false);
            SendInertialStatesToUAVCtrl.VisionCmdProc = _visionCmdProcess;
            SendInertialStatesToUAVCtrl.GeoCoordinateSys = _geoCoordinateSystem;

            _imageCaptureStatusMsg = new ImageCaptureStatusMsg();

            _visionProcessCmdStatusMsg = new VisionProcessingControlMsg();

            vehicleAndImageLocation1.GeoCoordinateSys = _geoCoordinateSystem;
            vehicleAndImageLocation1.Clear(); 


            //Head Tracking Display Tab
            headTrackingControlWidgt.VisionCmdProc = _visionCmdProcess;
            headTrackingControlWidgt.SetDefaultParameters();

            //displayImageInfo1

            //Vision System Response Message Processor / Handler...
            //Add Message handlers.
            //_visionBridge.VSResponseMsgProcessor.AddMessageHandler();

            //Vision System Video Stream Processor / Handler...
            //Add Message handlers.
            _visionBridge.VSVideoStreamMsgProcessor.AddMessageHandler("CompressedImageMsg", VSCompressedImageMsgHandler);


            //Vision System Monitor Message Processor / Handler...
            //Add Message handlers.
            _visionBridge.VSMonitorMsgProcessor.AddMessageHandler("VidereSystemStatusMsg", VidereSystemStatusMsgHandler);
            _visionBridge.VSMonitorMsgProcessor.AddMessageHandler("ManagerStatsMsg", VSManagerStatsMsgHandler);
            _visionBridge.VSMonitorMsgProcessor.AddMessageHandler("CameraCalStatsMsg", VSCameraCalStatusMsgHandler);
            _visionBridge.VSMonitorMsgProcessor.AddMessageHandler("FeatureMatchProcStatsMsg", VSFeatureMatchProcStatusMsgHandler);
            _visionBridge.VSMonitorMsgProcessor.AddMessageHandler("SteeringStatusMsg", DceEPASteeringStatusMsgHandler);
            _visionBridge.VSMonitorMsgProcessor.AddMessageHandler("ThrottlePositionStatusMsg", ThrottleStatusMsgHandler);
            _visionBridge.VSMonitorMsgProcessor.AddMessageHandler("BrakePositionStatusMsg", BrakeStatusMsgHandler);
            _visionBridge.VSMonitorMsgProcessor.AddMessageHandler("HeadOrientationMsg", HeadOrientationMsgHandler);
            _visionBridge.VSMonitorMsgProcessor.AddMessageHandler("SipAndPuffStatusMsg", SipAndPuffMsgHandler);
            _visionBridge.VSMonitorMsgProcessor.AddMessageHandler("IMUResponseMsg", IMUResponseMsgHandler);
            
            _visionBridge.VSResultsMsgProcessor.AddMessageHandler("TargetInformationMsg", VSTargetInfoMsgHandler);
            _visionBridge.VSResultsMsgProcessor.AddMessageHandler("ImageProcessingStatsMsg", VSImageProcessingStatsMsgHandler);

            //Vision System Results Messages arrive on a seperate pipe.
            //Add Message Handlers here
            _visionBridge.VSResultsMsgProcessor.AddMessageHandler("GPSFixMsg", VSGPSFixMsgHandler);
            _visionBridge.VSResultsMsgProcessor.AddMessageHandler("TrackHeadOrientationMsg", VSTrackHeadOrientationMsgHandler);
           
        }

        /// <summary>
        /// Clean-up when closing the form.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void FalconVisionViewerMainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            _visionBridge.DisconnectAllSockets();
        }


        private string ConnectVisionBridge(string visionSystemTpcAddr,
                                            string bridgeSystemTcpAddr)
        {
            string errorMsg = null;
            _visionBridge.VisionSystemConnectionType = BridgeConnectionType_e.tcp;
            _visionBridge.VisionSystemTCPAddrString = visionSystemTpcAddr;
            _visionBridge.BridgeSystemTCPAddrString = bridgeSystemTcpAddr;
            _visionBridge.VisionSystemCommandResponsePort = 5555;
            _visionBridge.VSResultsMsgProcessor.TCPPort = 5556;
            _visionBridge.VSVideoStreamMsgProcessor.TCPPort = 5557;
            _visionBridge.VSVideoStreamMsgProcessor.SocketHighWaterMark = 5;  //limit the number of images that can be kept in the queue.
            _visionBridge.VSMonitorMsgProcessor.TCPPort = 5559;
            _visionBridge.VSMonitorMsgProcessor.SocketHighWaterMark = 5;

            _visionBridge.MaxCmdResponseWaitTimeSeconds = 1.5;
            errorMsg = _visionBridge.ConnectToVisionSystemCommandResponseSocket();
            if (errorMsg != null)
            {
                MessageBox.Show("Error Connecting to VisionSystemRequestSocket: " + errorMsg);
                return errorMsg;
            }

            //HOPS and the FalconVision Monitor cannot both run on the same comuter
            //if binding to the same host IP address...  The ConnectPublishTelemeteryToVisionSystemSocket
            //is not really needed.
            //errorMsg = _visionBridge.ConnectPublishTelemeteryToVisionSystemSocket();
            if (errorMsg != null)
            {
                MessageBox.Show("Error Connecting to VisionSystemPublishSocket: " + errorMsg);
                _visionBridge.DisconnectAllSockets();
                return errorMsg;
            }

            //Vision System Response Message Processor / Handler...
            //Connect to the socket and check for connection errors.
            _visionBridge.VSResultsMsgProcessor.ClearErrors();
            _visionBridge.VSResultsMsgProcessor.ClearWarnings();
            _visionBridge.VSResultsMsgProcessor.ConnectToVisionSubscriptionSocketAsync();
            if (_visionBridge.VSResultsMsgProcessor.ProcessVisionMessageErrorCount > 0
                && _visionBridge.VSResultsMsgProcessor.ProcessVisionErrorMessage != null)
            {
                MessageBox.Show(_visionBridge.VSResultsMsgProcessor.ProcessVisionErrorMessage);
            }

            //Vision System Video Stream Message Processor / Handler...
            //Connect to the socket and check for connection errors.
            _visionBridge.VSVideoStreamMsgProcessor.ClearErrors();
            _visionBridge.VSVideoStreamMsgProcessor.ClearWarnings();
            _visionBridge.VSVideoStreamMsgProcessor.ConnectToVisionSubscriptionSocketAsync();
            if (_visionBridge.VSVideoStreamMsgProcessor.ProcessVisionMessageErrorCount > 0
                && _visionBridge.VSVideoStreamMsgProcessor.ProcessVisionErrorMessage != null)
            {
                MessageBox.Show(_visionBridge.VSVideoStreamMsgProcessor.ProcessVisionErrorMessage);
            }

            //Vision System Monitor Message Processor / Handler...
            //Connect to the socket and check for connection errors.
            _visionBridge.VSMonitorMsgProcessor.ClearErrors();
            _visionBridge.VSMonitorMsgProcessor.ClearWarnings();
            _visionBridge.VSMonitorMsgProcessor.ConnectToVisionSubscriptionSocketAsync();
            if (_visionBridge.VSMonitorMsgProcessor.ProcessVisionMessageErrorCount > 0
                && _visionBridge.VSMonitorMsgProcessor.ProcessVisionErrorMessage != null)
            {
                MessageBox.Show(_visionBridge.VSMonitorMsgProcessor.ProcessVisionErrorMessage);
            }

            _sendTelemetryToVisionSystem = true;
            //PublishStuffToVisionSystem();

            return errorMsg;
        }

       private string DisconnectVisionBridge()
        {
            string errorMsg = null;
            _sendTelemetryToVisionSystem = false;
            _visionBridge.DisconnectAllSockets();
            return errorMsg;
        }

       public void ResetAllVehicleActuators()
       {
           _brakeControlObj.ResetActuator();
           _throttleControlObj.ResetActuator();
           _steeringControlObj.resetSteeringControl();
           BrakePositionControlWidget.ReadAndDisplayActuatorValues();
           brakeLAPosCtrlSetupTab.ReadAndDisplayActuatorValues();
           throttleLAPosCtrlSetupTab.ReadAndDisplayActuatorValues();
           ThrottlePositionControlWidget.ReadAndDisplayActuatorValues();
           steeringControlWidget1.ReadAndDisplaySteeringControlValues();
       }

	    private bool TransmitSimpleMessage(string message)
	    {
            bool ok = false;
		    string respStr = _visionCmdProcess.SimpleCmdResponce(message);
		    tbVSMsgInfo.Text = respStr;
            if (respStr.StartsWith("OK"))
                ok = true;
            return ok;
	    }

        private void btnConnectToVS_Click(object sender, EventArgs e)
        {
            if (!_connectedToVisionBridge)
            {
                string visionSystemTcpAddr = tbVSIPAddr.Text;
                string bridgeSystemTcpAddr = tbHostIPAddr.Text;
                string errorMsg = ConnectVisionBridge(visionSystemTcpAddr, bridgeSystemTcpAddr);
                if (errorMsg != null)
                {
                    tbVSMsgInfo.Text = errorMsg;
                    btnConnectToVS.BackColor = System.Drawing.Color.Red;
                    _connectedToVisionBridge = false;
                }
                else
                {
                    tbVSMsgInfo.Text = "Connected To Vision System";
                    btnConnectToVS.BackColor = System.Drawing.Color.Green;
                    _connectedToVisionBridge = true;
                    timerMgrStatsUpdate.Enabled = true;
                }
            }
            else
            {
                //Disconnect from all Sockets.
                timerMgrStatsUpdate.Enabled = false;
                DisconnectVisionBridge();
                _connectedToVisionBridge = false;
                btnConnectToVS.BackColor = System.Drawing.Color.LightGray;
                tbVSMsgInfo.Text = "Disconnected from Vision System";
            }
        }

        private void btnGetVSInfo_Click(object sender, EventArgs e)
        {
            TransmitSimpleMessage("Info");
        }

        private void btnGetListOfManagers_Click(object sender, EventArgs e)
        {
            string[] listOfMgrNames = _visionCmdProcess.GetListOfManagerNames();
            MgrStatsCtrl_MgrNo_1.SetManagerNames(listOfMgrNames);
            MgrStatsCtrl_MgrNo_2.SetManagerNames(listOfMgrNames);
            MgrStatsCtrl_MgrNo_3.SetManagerNames(listOfMgrNames);
            MgrStatsCtrl_MgrNo_4.SetManagerNames(listOfMgrNames);
        }

        private void bnSSGPUProcessing_Click(object sender, EventArgs e)
        {
            if (_targetProcssingEnabled)
            {
                if (TransmitSimpleMessage("TargetProcessingDisable"))
                {
                    _targetProcssingEnabled = false;
                    bnSSGPUProcessing.BackColor = System.Drawing.Color.LightGray;
                }
                else
                {
                    bnSSGPUProcessing.BackColor = System.Drawing.Color.Yellow;
                }
            }
            else
            {
                if (TransmitSimpleMessage("TargetProcessingEnable"))
                {
                    _targetProcssingEnabled = true;
                    bnSSGPUProcessing.BackColor = System.Drawing.Color.Green;
                }
                else
                {
                    bnSSGPUProcessing.BackColor = System.Drawing.Color.Yellow;
                }
            }
            imageProcessControl1.TargetProcessingEnabled = _targetProcssingEnabled;
        }

        private void btnSSVideoStream_Click(object sender, EventArgs e)
        {
            if (_videoStreamEnabled)
            {
                if (TransmitSimpleMessage("StopStream"))
                {
                    _videoStreamEnabled = false;
                    btnSSVideoStream.BackColor = System.Drawing.Color.LightGray;
                }
                else
                {
                    btnSSVideoStream.BackColor = System.Drawing.Color.Yellow;
                }
            }
            else
            {
                if (TransmitSimpleMessage("StartStream"))
                {
                    _videoStreamEnabled = true;
                    btnSSVideoStream.BackColor = System.Drawing.Color.Green;
                }
                else
                {
                    btnSSVideoStream.BackColor = System.Drawing.Color.Yellow;
                }
            }
        }

        private void btnSSRecording_Click(object sender, EventArgs e)
        {
            if (_recordingEnabled)
            {
                if (TransmitSimpleMessage("StopRecording"))
                {
                    _recordingEnabled = false;
                    btnSSRecording.BackColor = System.Drawing.Color.LightGray;
                }
                else
                {
                    btnSSRecording.BackColor = System.Drawing.Color.Yellow;
                }
            }
            else
            {
                if (TransmitSimpleMessage("StartRecording"))
                {
                    _recordingEnabled = true;
                    btnSSRecording.BackColor = System.Drawing.Color.Green;
                }
                else
                {
                    btnSSRecording.BackColor = System.Drawing.Color.Yellow;
                }
            }
        }

        private void VSImageProcessingStatsMsgUpdateScreen(VisionProcessingControlMsg imgProcStatsMsg)
        {

            if (imgProcStatsMsg.StreamImagesEnabled)
            {
                _videoStreamEnabled = true;
                btnSSVideoStream.BackColor = System.Drawing.Color.Green;
            }
            else
            {
                _videoStreamEnabled = false;
                btnSSVideoStream.BackColor = System.Drawing.Color.LightGray;
            }

            if (imgProcStatsMsg.RecordImagesEnabled)
            {
                _recordingEnabled = true;
                btnSSRecording.BackColor = System.Drawing.Color.Green;
            }
            else
            {
                _recordingEnabled = false;
                btnSSRecording.BackColor = System.Drawing.Color.LightGray;
            }

            if (imgProcStatsMsg.TargetImageProcessingEnabled)
            {
                _targetProcssingEnabled = true;
                bnSSGPUProcessing.BackColor = System.Drawing.Color.Green;
            }
            else
            {
                _targetProcssingEnabled = false;
                bnSSGPUProcessing.BackColor = System.Drawing.Color.LightGray;
            }

        }


        private void btnShutdownVS_Click(object sender, EventArgs e)
        {
            TransmitSimpleMessage("Kill");
        }

        //Message Handlers


	    private void  VSCompressedImageMsgHandler(string image_params, byte[] image_data)
	    {
            if(tabMainScreen.SelectedTab == tabMainPage
                || tabMainScreen.SelectedTab == tabPgTargetInfo
                || tabMainScreen.SelectedTab == tabPgVehicleLocation )
            {
                ImageConverter converter = new ImageConverter();
                pictureBoxMainDsp.SizeMode = PictureBoxSizeMode.AutoSize;
                pictureBoxMainDsp.Image = (Image)converter.ConvertFrom(image_data);
            }
            else if(tabMainScreen.SelectedTab == tabCameraCal)
            {
                ImageConverter converter = new ImageConverter();
                pictureBoxCameraCalDisplay.SizeMode = PictureBoxSizeMode.AutoSize;
                pictureBoxCameraCalDisplay.Image = (Image)converter.ConvertFrom(image_data);
            }
            else if(tabMainScreen.SelectedTab == tabFeatureMatchProc)
            {
                ImageConverter converter = new ImageConverter();
                pictureBoxFeatureMatchProc.SizeMode = PictureBoxSizeMode.AutoSize;
                pictureBoxFeatureMatchProc.Image = (Image)converter.ConvertFrom(image_data);
            }
            else if(tabMainScreen.SelectedTab == tabPgTrackHead)
            {
                ImageConverter converter = new ImageConverter();
                pxBxTrackHeadDisplay.SizeMode = PictureBoxSizeMode.AutoSize;
                pxBxTrackHeadDisplay.Image = (Image)converter.ConvertFrom(image_data);
            }
	    }

        private void VSManagerStatsMsgHandler(string image_params, byte[] image_data)
        {
            ManagerStatsMsg mgrStats;
            mgrStats = ManagerStatsMsg.Deserialize(image_data);
            if (mgrStats != null 
                && tabMainScreen.SelectedTab == tabMgrStats )
            {
                if (mgrStats.ManagerName == MgrStatsCtrl_MgrNo_1.ManagerName)
                {
                    MgrStatsCtrl_MgrNo_1.SetMgrStats(mgrStats);
                }
                else if(mgrStats.ManagerName == MgrStatsCtrl_MgrNo_2.ManagerName)
                {
                    MgrStatsCtrl_MgrNo_2.SetMgrStats(mgrStats);
                }
                else if(mgrStats.ManagerName == MgrStatsCtrl_MgrNo_3.ManagerName)
                {
                    MgrStatsCtrl_MgrNo_3.SetMgrStats(mgrStats);
                }
                else if(mgrStats.ManagerName == MgrStatsCtrl_MgrNo_4.ManagerName)
                {
                    MgrStatsCtrl_MgrNo_4.SetMgrStats(mgrStats);
                }
            }
        }

        private void VidereSystemStatusMsgHandler(string image_params, byte[] msg_data)
        {
            VidereSystemControlPBMsg statusMsg;
            statusMsg = VidereSystemControlPBMsg.Deserialize(msg_data);
            if (statusMsg != null)
            {
                videreSystemStateControlForm.ProcessVidereSystemStateMsg(statusMsg);
            }
        }


        private void DceEPASteeringStatusMsgHandler(string image_params, byte[] msg_data)
        {
            DceEPASteeringStatusPBMsg statusMsg;
            statusMsg = DceEPASteeringStatusPBMsg.Deserialize(msg_data);
            if (statusMsg != null 
                && tabMainScreen.SelectedTab == tabCarControl )
            {
                _steeringControlObj.processSteeringStatusMsg(statusMsg);
                steeringControlWidget1.displaySteeringStatusInfo();
            }
        }

        private void ThrottleStatusMsgHandler(string image_params, byte[] image_data)
        {
            LinearActuatorPositionCtrlPBMsg statusMsg;
            statusMsg = LinearActuatorPositionCtrlPBMsg.Deserialize(image_data);
            if (statusMsg != null 
                && tabMainScreen.SelectedTab == tabCarControl )
            {
                _throttleControlObj.processPositionStatusMsg(statusMsg);
                ThrottlePositionControlWidget.displayPostionStatusInfo();
            }
            if (statusMsg != null 
                && tabMainScreen.SelectedTab == tabThrottleSetup )
            {
                _throttleControlObj.processPositionStatusMsg(statusMsg);
                throttleLAPosCtrlSetupTab.displayPostionStatusInfo();
            }
        }

        private void BrakeStatusMsgHandler(string image_params, byte[] image_data)
        {
            LinearActuatorPositionCtrlPBMsg statusMsg;
            statusMsg = LinearActuatorPositionCtrlPBMsg.Deserialize(image_data);
            if (statusMsg != null 
                && tabMainScreen.SelectedTab == tabCarControl )
            {
                _brakeControlObj.processPositionStatusMsg(statusMsg);
                BrakePositionControlWidget.displayPostionStatusInfo();
            }
            else if(statusMsg != null 
                && tabMainScreen.SelectedTab == tabBrakeSetup )
            {
                _brakeControlObj.processPositionStatusMsg(statusMsg);
                brakeLAPosCtrlSetupTab.displayPostionStatusInfo();
            }
        }

        private void HeadOrientationMsgHandler(string image_params, byte[] image_data)
        {
            HeadOrientationPBMsg statusMsg;
            statusMsg = HeadOrientationPBMsg.Deserialize(image_data);
            if (statusMsg != null 
                && tabMainScreen.SelectedTab == tabCarControl )
            {
                headOrientationSPMon.processHeadOrientationMsg(statusMsg);
            }
            else if (statusMsg != null 
                && tabMainScreen.SelectedTab == tabIMU )
            {
                headOrientationSPIMUTab.processHeadOrientationMsg(statusMsg);
            }
        }

        private void SipAndPuffMsgHandler(string image_params, byte[] image_data)
        {
            SipAndPuffPBMsg statusMsg;
            statusMsg = SipAndPuffPBMsg.Deserialize(image_data);
            if (statusMsg != null 
                && tabMainScreen.SelectedTab == tabCarControl )
            {
                headOrientationSPMon.processSipAndPuffMsg(statusMsg);
            }
        }

        private void IMUResponseMsgHandler(string image_params, byte[] image_data)
        {
            IMUCommandResponsePBMsg imuRspMsg;
            imuRspMsg = IMUCommandResponsePBMsg.Deserialize(image_data);
            if (imuRspMsg != null 
                && tabMainScreen.SelectedTab == tabIMU )
            {
                imuControlFrm.processIMUCmdResponseMsg(imuRspMsg);
            }
        }


        private void VSTargetInfoMsgHandler(string msgParams, byte[] msgData)
        {
            //Only handle the message if display is enabled... saves the overhead
            //of procesing the message if we don't care about it.
            ImageProcTargetInfoResultsMsg tgtInfoMsg;
            tgtInfoMsg = ImageProcTargetInfoResultsMsg.Deserialize(msgData);
            if (tgtInfoMsg != null)
            {

                if (tabMainScreen.SelectedTab == tabPgVehicleLocation )
                    //&& UavInertialStatesFromImageInfo.DisplayInertialStatesEnabled)
                {
                    UavInertialStatesFromImageInfo.SetInertialStates(tgtInfoMsg.VehicleInertialStates);
                    DisplayImageInfoForm.SetImageLocInfo(tgtInfoMsg.ImageLocation);
                }

                if (tabMainScreen.SelectedTab == tabPgTargetInfo)
                {
                    vehicleAndImageLocation1.UpdateLocationAndTargetInfo(tgtInfoMsg);
                }
            }
        }


        private void VSImageProcessingStatsMsgHandler(string msgParams, byte[] msgData)
        {
            VisionProcessingControlMsg imgProcStatsMsg;
            imgProcStatsMsg = VisionProcessingControlMsg.Deserialize(msgData);
            if (imgProcStatsMsg != null)
            {
                VSImageProcessingStatsMsgUpdateScreen(imgProcStatsMsg);
                imageProcessControl1.SetImageProcessCmdStatus(imgProcStatsMsg);
            }
        }

        private void VSCameraCalStatusMsgHandler(string msgParams, byte[] msgData)
        {
            //Only handle the message if display is enabled... saves the overhead
            //of procesing the message if we don't care about it.
            CameraCalStatusMsg cameraCalStatsMsg;
            cameraCalStatsMsg = CameraCalStatusMsg.Deserialize(msgData);
            if (cameraCalStatsMsg != null)
            {
                if (tabMainScreen.SelectedTab == tabCameraCal)
                {
                    //Pass message to Camera Cal Control Screen.
                    cameraCalControl1.processCameraCalStatusMessage(cameraCalStatsMsg);
                }
                else if (tabMainScreen.SelectedTab == tabPgTrackHead)
                {
                    headOrientationCalWidget1.processCameraCalStatusMessage(cameraCalStatsMsg);
                }
            }
        }

        private void VSFeatureMatchProcStatusMsgHandler(string msgParams, byte[] msgData)
        {
            //Only handle the message if display is enabled... saves the overhead
            //of procesing the message if we don't care about it.
            FeatureMatchProcStatusPBMsg fmpStatsMsg;
            fmpStatsMsg = FeatureMatchProcStatusPBMsg.Deserialize(msgData);
            if (fmpStatsMsg != null)
            {
                //Pass message to Camera Cal Control Screen.
                featureMatchProcessControl1.processFeatureMatchlStatusMessage(fmpStatsMsg);
                processTimerStatus1.ProcessImageFeatureMatchStatusMsg(fmpStatsMsg);
            }
        }

        private void VSGPSFixMsgHandler(string msgParams, byte[] msgData)
        {
            //Only handle the message if display is enabled... saves the overhead
            //of procesing the message if we don't care about it.
            GPSFixPBMsg gpsFixMsg;
            gpsFixMsg = GPSFixPBMsg.Deserialize(msgData);
            if (gpsFixMsg != null)
            {
                gpsFixWidget1.processGPSFix(gpsFixMsg);
            }
        }

        private void VSTrackHeadOrientationMsgHandler(string msgParams, byte[] msgData)
        {
            //Only handle the message if display is enabled... saves the overhead
            //of procesing the message if we don't care about it.
            TrackHeadOrientationPBMsg trackHeadOrientationMsg;
            trackHeadOrientationMsg = TrackHeadOrientationPBMsg.Deserialize(msgData);
            if (trackHeadOrientationMsg != null)
            {
                //Pass message to Camera Cal Control Screen.
                //featureMatchProcessControl1.processFeatureMatchlStatusMessage(fmpStatsMsg);
                //processTimerStatus1.ProcessImageFeatureMatchStatusMsg(fmpStatsMsg);
            }
        }


        private void timerMgrStatsUpdate_Tick(object sender, EventArgs e)
        {
            if (_connectedToVisionBridge)
            {
                string cmdResponseMsg;
                try
                {
                    //Get the Image Capture Status
                    _imageCaptureStatusMsg = _visionCmdProcess.GetImageCaptureStatus(out cmdResponseMsg);
                    if (_imageCaptureStatusMsg != null)
                    {
                        imageCaptureControl1.ProcessImageCaptureStatusMsg(_imageCaptureStatusMsg);
                        imageCaptureSetupAndStatus1.ProcessImageCaptureStatusMsg(_imageCaptureStatusMsg);
                    }

                    _visionProcessCmdStatusMsg = _visionCmdProcess.GetVisionProcessControlStatus(out cmdResponseMsg);
                    if (_visionProcessCmdStatusMsg != null)
                    {
                        VSImageProcessingStatsMsgUpdateScreen(_visionProcessCmdStatusMsg);
                        imageProcessControl1.SetImageProcessCmdStatus(_visionProcessCmdStatusMsg);
                    }

                    //Get Manager Stats... this is a temp location for handling this;
                    /*********************  Now Sent by the Vision system Automatically ***
                    int statsNo = MgrStatsCounter % 4;
                    if (statsNo == 0)
                    {
                        string resultsMsg;
                        ManagerStatsMsg mgrStats = _visionCmdProcess.GetManagerStats("ImageCaptureManager", out resultsMsg);
                        MgrStatsCtrl_ImageCapture.SetMgrStats(mgrStats);
                    }
                    else if (statsNo == 1)
                    {
                        string resultsMsg;
                        ManagerStatsMsg mgrStats = _visionCmdProcess.GetManagerStats("VisionProcessManager", out resultsMsg);
                        MgrStatsCtrl_GpuVisionProcess.SetMgrStats(mgrStats);
                    }
                    else if (statsNo == 2)
                    {
                        string resultsMsg;
                        ManagerStatsMsg mgrStats = _visionCmdProcess.GetManagerStats("StreamRecordManager", out resultsMsg);
                        MgrStatsCtrl_StreamRecord.SetMgrStats(mgrStats);
                    }
                    else if (statsNo == 3)
                    {
                        string resultsMsg;
                        ManagerStatsMsg mgrStats = _visionCmdProcess.GetManagerStats("CommsManager", out resultsMsg);
                        MgrStatsCtrl_Comm.SetMgrStats(mgrStats);
                    }

                    ++MgrStatsCounter;
                     * *************************************/
                }
                catch (Exception ex)
                {
                    MessageBox.Show("MgrStatsUpdate Exception: " + ex.Message);
                }
            }
        }

        private void btnCopyInertialStates_Click(object sender, EventArgs e)
        {
            VehicleInterialStatesMsg visMsg = UavInertialStatesFromImageInfo.ReadInertialStates(true);
            SendInertialStatesToUAVCtrl.SetInertialStates(visMsg);
        }

        private void imageProcessControl1_Load(object sender, EventArgs e)
        {

        }



        /*************************************************
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
				    _bridge.PublishTelemetry(msg);

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
				    
				    ++cntr;
			    }
			    catch(Exception ex)
			    {
				    Console.Write("Exception: " + ex.Message);
			    }
			    await Task.Delay(1000);
		    }
	    }
        *************************************************/


    }
}
