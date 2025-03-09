namespace FalconVisionMonitorViewer
{
    partial class FalconVisionViewerMainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.tabMainScreen = new System.Windows.Forms.TabControl();
            this.tabCarControl = new System.Windows.Forms.TabPage();
            this.gpsFixWidget1 = new FalconVisionMonitorViewer.GuiFormsWidgets.GPSFixWidget();
            this.vehicleControlParametersWidget1 = new FalconVisionMonitorViewer.GuiFormsWidgets.VehicleControlParametersWidget();
            this.headOrientationSPMon = new FalconVisionMonitorViewer.GuiFormsWidgets.HeadOrientationSPWidget();
            this.BrakePositionControlWidget = new CarCANBusMonitor.Widgets.LinearActuatorPositionControl();
            this.ThrottlePositionControlWidget = new CarCANBusMonitor.Widgets.LinearActuatorPositionControl();
            this.steeringControlWidget1 = new CarCANBusMonitor.Widgets.SteeringControlWidget();
            this.tabIMU = new System.Windows.Forms.TabPage();
            this.imuControlFrm = new FalconVisionMonitorViewer.GuiFormsWidgets.IMUControlWidget();
            this.headOrientationSPIMUTab = new FalconVisionMonitorViewer.GuiFormsWidgets.HeadOrientationSPWidget();
            this.tabMgrStats = new System.Windows.Forms.TabPage();
            this.btnGetListOfManagers = new System.Windows.Forms.Button();
            this.MgrStatsCtrl_MgrNo_4 = new FalconVisionMonitorViewer.GuiFormsWidgets.ManagerStatsUserControl();
            this.MgrStatsCtrl_MgrNo_3 = new FalconVisionMonitorViewer.GuiFormsWidgets.ManagerStatsUserControl();
            this.MgrStatsCtrl_MgrNo_2 = new FalconVisionMonitorViewer.GuiFormsWidgets.ManagerStatsUserControl();
            this.MgrStatsCtrl_MgrNo_1 = new FalconVisionMonitorViewer.GuiFormsWidgets.ManagerStatsUserControl();
            this.tabPgVehicleLocation = new System.Windows.Forms.TabPage();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.btnCopyInertialStates = new System.Windows.Forms.Button();
            this.lbl_ISBottom2 = new System.Windows.Forms.Label();
            this.lbl_ISBottom1 = new System.Windows.Forms.Label();
            this.lbl_ISTop = new System.Windows.Forms.Label();
            this.SendInertialStatesToUAVCtrl = new FalconVisionMonitorViewer.GuiFormsWidgets.UAVInertialStates();
            this.DisplayImageInfoForm = new FalconVisionMonitorViewer.GuiFormsWidgets.DisplayImageInfo();
            this.UavInertialStatesFromImageInfo = new FalconVisionMonitorViewer.GuiFormsWidgets.UAVInertialStates();
            this.tabPgSetup = new System.Windows.Forms.TabPage();
            this.streamRecordControlWidget1 = new FalconVisionMonitorViewer.GuiFormsWidgets.StreamRecordControlWidget();
            this.cameraParametersSetupWidget1 = new FalconVisionMonitorViewer.GuiFormsWidgets.CameraParametersSetupWidget();
            this.cameraOrientationControl1 = new FalconVisionMonitorViewer.GuiFormsWidgets.CameraOrientationControl();
            this.imageProcessControl1 = new FalconVisionMonitorViewer.GuiFormsWidgets.ImageProcessControl();
            this.imageCaptureSetupAndStatus1 = new FalconVisionMonitorViewer.GuiFormsWidgets.ImageCaptureSetupAndStatus();
            this.geoCoordinateSystemSetup1 = new FalconVisionMonitorViewer.GuiFormsWidgets.GeoCoordinateSystemSetup();
            this.tabMainPage = new System.Windows.Forms.TabPage();
            this.panelVideoImage1 = new System.Windows.Forms.Panel();
            this.pictureBoxMainDsp = new System.Windows.Forms.PictureBox();
            this.tabPgTargetInfo = new System.Windows.Forms.TabPage();
            this.blobDetectorParameters1 = new FalconVisionMonitorViewer.GuiFormsWidgets.BlobDetectorParameters();
            this.vehicleAndImageLocation1 = new FalconVisionMonitorViewer.GuiFormsWidgets.VehicleAndImageLocation();
            this.tabCameraCal = new System.Windows.Forms.TabPage();
            this.cameraMountCorrectionInput1 = new FalconVisionMonitorViewer.GuiFormsWidgets.CameraMountCorrectionInput();
            this.cameraCalChessBdInput1 = new FalconVisionMonitorViewer.GuiFormsWidgets.CameraCalChessBdInput();
            this.pnlCalImage = new System.Windows.Forms.Panel();
            this.pictureBoxCameraCalDisplay = new System.Windows.Forms.PictureBox();
            this.cameraCalControl1 = new FalconVisionMonitorViewer.GuiFormsWidgets.CameraCalControl();
            this.tabFeatureMatchProc = new System.Windows.Forms.TabPage();
            this.processTimerStatus1 = new FalconVisionMonitorViewer.GuiFormsWidgets.ProcessTimerStatus();
            this.pictureBoxFeatureMatchProc = new System.Windows.Forms.PictureBox();
            this.featureMatchProcessControl1 = new FalconVisionMonitorViewer.GuiFormsWidgets.FeatureMatchProcessControl();
            this.tabBrakeSetup = new System.Windows.Forms.TabPage();
            this.brakeLAPosCtrlSetupTab = new CarCANBusMonitor.Widgets.LinearActuatorPositionControl();
            this.brakeLAConfigSetupTab = new CarCANBusMonitor.Widgets.KarTechLinearActuatorSetupWidget();
            this.tabThrottleSetup = new System.Windows.Forms.TabPage();
            this.throttleLAConfigSetupTab = new CarCANBusMonitor.Widgets.KarTechLinearActuatorSetupWidget();
            this.throttleLAPosCtrlSetupTab = new CarCANBusMonitor.Widgets.LinearActuatorPositionControl();
            this.tabPgTrackHead = new System.Windows.Forms.TabPage();
            this.headOrientationControlWidget1 = new FalconVisionMonitorViewer.GuiFormsWidgets.HeadOrientationControlWidget();
            this.headOrientationCalWidget1 = new FalconVisionMonitorViewer.GuiFormsWidgets.HeadOrientationCalWidget();
            this.pxBxTrackHeadDisplay = new System.Windows.Forms.PictureBox();
            this.headTrackingControlWidgt = new FalconVisionMonitorViewer.GuiFormsWidgets.HeadTrackingControlWidget();
            this.btnConnectToVS = new System.Windows.Forms.Button();
            this.tbVSIPAddr = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.tbHostIPAddr = new System.Windows.Forms.TextBox();
            this.tbVSMsgInfo = new System.Windows.Forms.TextBox();
            this.bnSSGPUProcessing = new System.Windows.Forms.Button();
            this.btnSSVideoStream = new System.Windows.Forms.Button();
            this.btnSSRecording = new System.Windows.Forms.Button();
            this.btnShutdownVS = new System.Windows.Forms.Button();
            this.btnGetVSInfo = new System.Windows.Forms.Button();
            this.timerMgrStatsUpdate = new System.Windows.Forms.Timer(this.components);
            this.videreSystemStateControlForm = new FalconVisionMonitorViewer.GuiFormsWidgets.VidereSystemStateControlWidget();
            this.imageCaptureControl1 = new FalconVisionMonitorViewer.GuiFormsWidgets.ImageCaptureControl();
            this.tabMainScreen.SuspendLayout();
            this.tabCarControl.SuspendLayout();
            this.tabIMU.SuspendLayout();
            this.tabMgrStats.SuspendLayout();
            this.tabPgVehicleLocation.SuspendLayout();
            this.tabPgSetup.SuspendLayout();
            this.tabMainPage.SuspendLayout();
            this.panelVideoImage1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxMainDsp)).BeginInit();
            this.tabPgTargetInfo.SuspendLayout();
            this.tabCameraCal.SuspendLayout();
            this.pnlCalImage.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxCameraCalDisplay)).BeginInit();
            this.tabFeatureMatchProc.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxFeatureMatchProc)).BeginInit();
            this.tabBrakeSetup.SuspendLayout();
            this.tabThrottleSetup.SuspendLayout();
            this.tabPgTrackHead.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pxBxTrackHeadDisplay)).BeginInit();
            this.SuspendLayout();
            // 
            // menuStrip1
            // 
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(1254, 24);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // tabMainScreen
            // 
            this.tabMainScreen.Controls.Add(this.tabCarControl);
            this.tabMainScreen.Controls.Add(this.tabIMU);
            this.tabMainScreen.Controls.Add(this.tabMgrStats);
            this.tabMainScreen.Controls.Add(this.tabPgVehicleLocation);
            this.tabMainScreen.Controls.Add(this.tabPgSetup);
            this.tabMainScreen.Controls.Add(this.tabMainPage);
            this.tabMainScreen.Controls.Add(this.tabPgTargetInfo);
            this.tabMainScreen.Controls.Add(this.tabCameraCal);
            this.tabMainScreen.Controls.Add(this.tabFeatureMatchProc);
            this.tabMainScreen.Controls.Add(this.tabBrakeSetup);
            this.tabMainScreen.Controls.Add(this.tabThrottleSetup);
            this.tabMainScreen.Controls.Add(this.tabPgTrackHead);
            this.tabMainScreen.Location = new System.Drawing.Point(155, 0);
            this.tabMainScreen.Name = "tabMainScreen";
            this.tabMainScreen.SelectedIndex = 0;
            this.tabMainScreen.Size = new System.Drawing.Size(1087, 704);
            this.tabMainScreen.TabIndex = 1;
            // 
            // tabCarControl
            // 
            this.tabCarControl.Controls.Add(this.gpsFixWidget1);
            this.tabCarControl.Controls.Add(this.vehicleControlParametersWidget1);
            this.tabCarControl.Controls.Add(this.headOrientationSPMon);
            this.tabCarControl.Controls.Add(this.BrakePositionControlWidget);
            this.tabCarControl.Controls.Add(this.ThrottlePositionControlWidget);
            this.tabCarControl.Controls.Add(this.steeringControlWidget1);
            this.tabCarControl.Location = new System.Drawing.Point(4, 22);
            this.tabCarControl.Name = "tabCarControl";
            this.tabCarControl.Padding = new System.Windows.Forms.Padding(3);
            this.tabCarControl.Size = new System.Drawing.Size(1079, 678);
            this.tabCarControl.TabIndex = 7;
            this.tabCarControl.Text = "Car Control";
            this.tabCarControl.UseVisualStyleBackColor = true;
            // 
            // gpsFixWidget1
            // 
            this.gpsFixWidget1.Location = new System.Drawing.Point(8, 249);
            this.gpsFixWidget1.Name = "gpsFixWidget1";
            this.gpsFixWidget1.Size = new System.Drawing.Size(327, 160);
            this.gpsFixWidget1.TabIndex = 5;
            // 
            // vehicleControlParametersWidget1
            // 
            this.vehicleControlParametersWidget1.Location = new System.Drawing.Point(744, 6);
            this.vehicleControlParametersWidget1.Name = "vehicleControlParametersWidget1";
            this.vehicleControlParametersWidget1.Size = new System.Drawing.Size(246, 614);
            this.vehicleControlParametersWidget1.TabIndex = 4;
            // 
            // headOrientationSPMon
            // 
            this.headOrientationSPMon.Location = new System.Drawing.Point(400, 6);
            this.headOrientationSPMon.Name = "headOrientationSPMon";
            this.headOrientationSPMon.Size = new System.Drawing.Size(325, 329);
            this.headOrientationSPMon.TabIndex = 3;
            // 
            // BrakePositionControlWidget
            // 
            this.BrakePositionControlWidget.Location = new System.Drawing.Point(375, 415);
            this.BrakePositionControlWidget.Name = "BrakePositionControlWidget";
            this.BrakePositionControlWidget.Size = new System.Drawing.Size(350, 231);
            this.BrakePositionControlWidget.TabIndex = 2;
            // 
            // ThrottlePositionControlWidget
            // 
            this.ThrottlePositionControlWidget.Location = new System.Drawing.Point(8, 415);
            this.ThrottlePositionControlWidget.Name = "ThrottlePositionControlWidget";
            this.ThrottlePositionControlWidget.Size = new System.Drawing.Size(350, 235);
            this.ThrottlePositionControlWidget.TabIndex = 1;
            // 
            // steeringControlWidget1
            // 
            this.steeringControlWidget1.Location = new System.Drawing.Point(8, 6);
            this.steeringControlWidget1.Name = "steeringControlWidget1";
            this.steeringControlWidget1.Size = new System.Drawing.Size(373, 237);
            this.steeringControlWidget1.TabIndex = 0;
            // 
            // tabIMU
            // 
            this.tabIMU.Controls.Add(this.imuControlFrm);
            this.tabIMU.Controls.Add(this.headOrientationSPIMUTab);
            this.tabIMU.Location = new System.Drawing.Point(4, 22);
            this.tabIMU.Name = "tabIMU";
            this.tabIMU.Padding = new System.Windows.Forms.Padding(3);
            this.tabIMU.Size = new System.Drawing.Size(1079, 678);
            this.tabIMU.TabIndex = 8;
            this.tabIMU.Text = "IMU ";
            this.tabIMU.UseVisualStyleBackColor = true;
            // 
            // imuControlFrm
            // 
            this.imuControlFrm.Location = new System.Drawing.Point(11, 24);
            this.imuControlFrm.Name = "imuControlFrm";
            this.imuControlFrm.Size = new System.Drawing.Size(360, 142);
            this.imuControlFrm.TabIndex = 1;
            // 
            // headOrientationSPIMUTab
            // 
            this.headOrientationSPIMUTab.Location = new System.Drawing.Point(445, 24);
            this.headOrientationSPIMUTab.Name = "headOrientationSPIMUTab";
            this.headOrientationSPIMUTab.Size = new System.Drawing.Size(325, 329);
            this.headOrientationSPIMUTab.TabIndex = 0;
            // 
            // tabMgrStats
            // 
            this.tabMgrStats.Controls.Add(this.btnGetListOfManagers);
            this.tabMgrStats.Controls.Add(this.MgrStatsCtrl_MgrNo_4);
            this.tabMgrStats.Controls.Add(this.MgrStatsCtrl_MgrNo_3);
            this.tabMgrStats.Controls.Add(this.MgrStatsCtrl_MgrNo_2);
            this.tabMgrStats.Controls.Add(this.MgrStatsCtrl_MgrNo_1);
            this.tabMgrStats.Location = new System.Drawing.Point(4, 22);
            this.tabMgrStats.Name = "tabMgrStats";
            this.tabMgrStats.Padding = new System.Windows.Forms.Padding(3);
            this.tabMgrStats.Size = new System.Drawing.Size(1079, 678);
            this.tabMgrStats.TabIndex = 1;
            this.tabMgrStats.Text = "Mgr Stats";
            this.tabMgrStats.UseVisualStyleBackColor = true;
            // 
            // btnGetListOfManagers
            // 
            this.btnGetListOfManagers.Location = new System.Drawing.Point(694, 562);
            this.btnGetListOfManagers.Name = "btnGetListOfManagers";
            this.btnGetListOfManagers.Size = new System.Drawing.Size(149, 23);
            this.btnGetListOfManagers.TabIndex = 4;
            this.btnGetListOfManagers.Text = "Get List Of Managers";
            this.btnGetListOfManagers.UseVisualStyleBackColor = true;
            this.btnGetListOfManagers.Click += new System.EventHandler(this.btnGetListOfManagers_Click);
            // 
            // MgrStatsCtrl_MgrNo_4
            // 
            this.MgrStatsCtrl_MgrNo_4.Location = new System.Drawing.Point(426, 279);
            this.MgrStatsCtrl_MgrNo_4.Name = "MgrStatsCtrl_MgrNo_4";
            this.MgrStatsCtrl_MgrNo_4.Size = new System.Drawing.Size(420, 277);
            this.MgrStatsCtrl_MgrNo_4.TabIndex = 3;
            // 
            // MgrStatsCtrl_MgrNo_3
            // 
            this.MgrStatsCtrl_MgrNo_3.Location = new System.Drawing.Point(6, 279);
            this.MgrStatsCtrl_MgrNo_3.Name = "MgrStatsCtrl_MgrNo_3";
            this.MgrStatsCtrl_MgrNo_3.Size = new System.Drawing.Size(420, 277);
            this.MgrStatsCtrl_MgrNo_3.TabIndex = 2;
            // 
            // MgrStatsCtrl_MgrNo_2
            // 
            this.MgrStatsCtrl_MgrNo_2.Location = new System.Drawing.Point(423, 6);
            this.MgrStatsCtrl_MgrNo_2.Name = "MgrStatsCtrl_MgrNo_2";
            this.MgrStatsCtrl_MgrNo_2.Size = new System.Drawing.Size(420, 277);
            this.MgrStatsCtrl_MgrNo_2.TabIndex = 1;
            // 
            // MgrStatsCtrl_MgrNo_1
            // 
            this.MgrStatsCtrl_MgrNo_1.Location = new System.Drawing.Point(6, 6);
            this.MgrStatsCtrl_MgrNo_1.Name = "MgrStatsCtrl_MgrNo_1";
            this.MgrStatsCtrl_MgrNo_1.Size = new System.Drawing.Size(420, 277);
            this.MgrStatsCtrl_MgrNo_1.TabIndex = 0;
            // 
            // tabPgVehicleLocation
            // 
            this.tabPgVehicleLocation.Controls.Add(this.label5);
            this.tabPgVehicleLocation.Controls.Add(this.label4);
            this.tabPgVehicleLocation.Controls.Add(this.btnCopyInertialStates);
            this.tabPgVehicleLocation.Controls.Add(this.lbl_ISBottom2);
            this.tabPgVehicleLocation.Controls.Add(this.lbl_ISBottom1);
            this.tabPgVehicleLocation.Controls.Add(this.lbl_ISTop);
            this.tabPgVehicleLocation.Controls.Add(this.SendInertialStatesToUAVCtrl);
            this.tabPgVehicleLocation.Controls.Add(this.DisplayImageInfoForm);
            this.tabPgVehicleLocation.Controls.Add(this.UavInertialStatesFromImageInfo);
            this.tabPgVehicleLocation.Location = new System.Drawing.Point(4, 22);
            this.tabPgVehicleLocation.Name = "tabPgVehicleLocation";
            this.tabPgVehicleLocation.Padding = new System.Windows.Forms.Padding(3);
            this.tabPgVehicleLocation.Size = new System.Drawing.Size(1079, 678);
            this.tabPgVehicleLocation.TabIndex = 3;
            this.tabPgVehicleLocation.Text = "UAV Image Loc";
            this.tabPgVehicleLocation.UseVisualStyleBackColor = true;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(479, 373);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(52, 13);
            this.label5.TabIndex = 8;
            this.label5.Text = "from UAV";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(479, 357);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(98, 13);
            this.label4.TabIndex = 7;
            this.label4.Text = "Copy Inertial States";
            // 
            // btnCopyInertialStates
            // 
            this.btnCopyInertialStates.Location = new System.Drawing.Point(482, 389);
            this.btnCopyInertialStates.Name = "btnCopyInertialStates";
            this.btnCopyInertialStates.Size = new System.Drawing.Size(75, 23);
            this.btnCopyInertialStates.TabIndex = 6;
            this.btnCopyInertialStates.Text = "Copy";
            this.btnCopyInertialStates.UseVisualStyleBackColor = true;
            this.btnCopyInertialStates.Click += new System.EventHandler(this.btnCopyInertialStates_Click);
            // 
            // lbl_ISBottom2
            // 
            this.lbl_ISBottom2.AutoSize = true;
            this.lbl_ISBottom2.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lbl_ISBottom2.Location = new System.Drawing.Point(479, 325);
            this.lbl_ISBottom2.Name = "lbl_ISBottom2";
            this.lbl_ISBottom2.Size = new System.Drawing.Size(86, 17);
            this.lbl_ISBottom2.TabIndex = 5;
            this.lbl_ISBottom2.Text = "to the UAV";
            this.lbl_ISBottom2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // lbl_ISBottom1
            // 
            this.lbl_ISBottom1.AutoSize = true;
            this.lbl_ISBottom1.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lbl_ISBottom1.Location = new System.Drawing.Point(479, 308);
            this.lbl_ISBottom1.Name = "lbl_ISBottom1";
            this.lbl_ISBottom1.Size = new System.Drawing.Size(151, 17);
            this.lbl_ISBottom1.TabIndex = 4;
            this.lbl_ISBottom1.Text = "Send Inertial States";
            this.lbl_ISBottom1.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // lbl_ISTop
            // 
            this.lbl_ISTop.AutoSize = true;
            this.lbl_ISTop.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lbl_ISTop.Location = new System.Drawing.Point(479, 21);
            this.lbl_ISTop.Name = "lbl_ISTop";
            this.lbl_ISTop.Size = new System.Drawing.Size(182, 17);
            this.lbl_ISTop.TabIndex = 3;
            this.lbl_ISTop.Text = "Inertial States from UAV";
            this.lbl_ISTop.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // SendInertialStatesToUAVCtrl
            // 
            this.SendInertialStatesToUAVCtrl.Location = new System.Drawing.Point(6, 297);
            this.SendInertialStatesToUAVCtrl.Name = "SendInertialStatesToUAVCtrl";
            this.SendInertialStatesToUAVCtrl.Size = new System.Drawing.Size(467, 299);
            this.SendInertialStatesToUAVCtrl.TabIndex = 2;
            // 
            // DisplayImageInfoForm
            // 
            this.DisplayImageInfoForm.Location = new System.Drawing.Point(466, 54);
            this.DisplayImageInfoForm.Name = "DisplayImageInfoForm";
            this.DisplayImageInfoForm.Size = new System.Drawing.Size(204, 237);
            this.DisplayImageInfoForm.TabIndex = 1;
            // 
            // UavInertialStatesFromImageInfo
            // 
            this.UavInertialStatesFromImageInfo.Location = new System.Drawing.Point(6, 6);
            this.UavInertialStatesFromImageInfo.Name = "UavInertialStatesFromImageInfo";
            this.UavInertialStatesFromImageInfo.Size = new System.Drawing.Size(467, 299);
            this.UavInertialStatesFromImageInfo.TabIndex = 0;
            // 
            // tabPgSetup
            // 
            this.tabPgSetup.Controls.Add(this.streamRecordControlWidget1);
            this.tabPgSetup.Controls.Add(this.cameraParametersSetupWidget1);
            this.tabPgSetup.Controls.Add(this.cameraOrientationControl1);
            this.tabPgSetup.Controls.Add(this.imageProcessControl1);
            this.tabPgSetup.Controls.Add(this.imageCaptureSetupAndStatus1);
            this.tabPgSetup.Controls.Add(this.geoCoordinateSystemSetup1);
            this.tabPgSetup.Location = new System.Drawing.Point(4, 22);
            this.tabPgSetup.Name = "tabPgSetup";
            this.tabPgSetup.Padding = new System.Windows.Forms.Padding(3);
            this.tabPgSetup.Size = new System.Drawing.Size(1079, 678);
            this.tabPgSetup.TabIndex = 2;
            this.tabPgSetup.Text = "Setup";
            this.tabPgSetup.UseVisualStyleBackColor = true;
            // 
            // streamRecordControlWidget1
            // 
            this.streamRecordControlWidget1.Location = new System.Drawing.Point(403, 411);
            this.streamRecordControlWidget1.Name = "streamRecordControlWidget1";
            this.streamRecordControlWidget1.Size = new System.Drawing.Size(242, 160);
            this.streamRecordControlWidget1.TabIndex = 5;
            // 
            // cameraParametersSetupWidget1
            // 
            this.cameraParametersSetupWidget1.Location = new System.Drawing.Point(667, 413);
            this.cameraParametersSetupWidget1.Name = "cameraParametersSetupWidget1";
            this.cameraParametersSetupWidget1.Size = new System.Drawing.Size(254, 158);
            this.cameraParametersSetupWidget1.TabIndex = 4;
            this.cameraParametersSetupWidget1.Visible = false;
            // 
            // cameraOrientationControl1
            // 
            this.cameraOrientationControl1.Location = new System.Drawing.Point(403, 228);
            this.cameraOrientationControl1.Name = "cameraOrientationControl1";
            this.cameraOrientationControl1.Size = new System.Drawing.Size(304, 177);
            this.cameraOrientationControl1.TabIndex = 3;
            // 
            // imageProcessControl1
            // 
            this.imageProcessControl1.Location = new System.Drawing.Point(11, 341);
            this.imageProcessControl1.Name = "imageProcessControl1";
            this.imageProcessControl1.Size = new System.Drawing.Size(364, 222);
            this.imageProcessControl1.TabIndex = 2;
            this.imageProcessControl1.Load += new System.EventHandler(this.imageProcessControl1_Load);
            // 
            // imageCaptureSetupAndStatus1
            // 
            this.imageCaptureSetupAndStatus1.Location = new System.Drawing.Point(11, 6);
            this.imageCaptureSetupAndStatus1.Name = "imageCaptureSetupAndStatus1";
            this.imageCaptureSetupAndStatus1.Size = new System.Drawing.Size(364, 329);
            this.imageCaptureSetupAndStatus1.TabIndex = 1;
            // 
            // geoCoordinateSystemSetup1
            // 
            this.geoCoordinateSystemSetup1.Location = new System.Drawing.Point(403, 6);
            this.geoCoordinateSystemSetup1.Name = "geoCoordinateSystemSetup1";
            this.geoCoordinateSystemSetup1.Size = new System.Drawing.Size(393, 233);
            this.geoCoordinateSystemSetup1.TabIndex = 0;
            // 
            // tabMainPage
            // 
            this.tabMainPage.Controls.Add(this.panelVideoImage1);
            this.tabMainPage.Location = new System.Drawing.Point(4, 22);
            this.tabMainPage.Name = "tabMainPage";
            this.tabMainPage.Padding = new System.Windows.Forms.Padding(3);
            this.tabMainPage.Size = new System.Drawing.Size(1079, 678);
            this.tabMainPage.TabIndex = 0;
            this.tabMainPage.Text = "Camera View";
            this.tabMainPage.UseVisualStyleBackColor = true;
            // 
            // panelVideoImage1
            // 
            this.panelVideoImage1.AutoScroll = true;
            this.panelVideoImage1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelVideoImage1.Controls.Add(this.pictureBoxMainDsp);
            this.panelVideoImage1.Location = new System.Drawing.Point(6, 3);
            this.panelVideoImage1.Name = "panelVideoImage1";
            this.panelVideoImage1.Size = new System.Drawing.Size(1067, 672);
            this.panelVideoImage1.TabIndex = 1;
            // 
            // pictureBoxMainDsp
            // 
            this.pictureBoxMainDsp.Location = new System.Drawing.Point(-4, 3);
            this.pictureBoxMainDsp.Name = "pictureBoxMainDsp";
            this.pictureBoxMainDsp.Size = new System.Drawing.Size(1080, 668);
            this.pictureBoxMainDsp.TabIndex = 0;
            this.pictureBoxMainDsp.TabStop = false;
            // 
            // tabPgTargetInfo
            // 
            this.tabPgTargetInfo.Controls.Add(this.blobDetectorParameters1);
            this.tabPgTargetInfo.Controls.Add(this.vehicleAndImageLocation1);
            this.tabPgTargetInfo.Location = new System.Drawing.Point(4, 22);
            this.tabPgTargetInfo.Name = "tabPgTargetInfo";
            this.tabPgTargetInfo.Padding = new System.Windows.Forms.Padding(3);
            this.tabPgTargetInfo.Size = new System.Drawing.Size(1079, 678);
            this.tabPgTargetInfo.TabIndex = 4;
            this.tabPgTargetInfo.Text = "TargetInfo";
            this.tabPgTargetInfo.UseVisualStyleBackColor = true;
            // 
            // blobDetectorParameters1
            // 
            this.blobDetectorParameters1.Location = new System.Drawing.Point(620, 16);
            this.blobDetectorParameters1.Name = "blobDetectorParameters1";
            this.blobDetectorParameters1.Size = new System.Drawing.Size(214, 529);
            this.blobDetectorParameters1.TabIndex = 1;
            // 
            // vehicleAndImageLocation1
            // 
            this.vehicleAndImageLocation1.FreezeUpdate = false;
            this.vehicleAndImageLocation1.GeoCoordinateSys = null;
            this.vehicleAndImageLocation1.Location = new System.Drawing.Point(7, 7);
            this.vehicleAndImageLocation1.Name = "vehicleAndImageLocation1";
            this.vehicleAndImageLocation1.Size = new System.Drawing.Size(607, 574);
            this.vehicleAndImageLocation1.TabIndex = 0;
            // 
            // tabCameraCal
            // 
            this.tabCameraCal.Controls.Add(this.cameraMountCorrectionInput1);
            this.tabCameraCal.Controls.Add(this.cameraCalChessBdInput1);
            this.tabCameraCal.Controls.Add(this.pnlCalImage);
            this.tabCameraCal.Controls.Add(this.cameraCalControl1);
            this.tabCameraCal.Location = new System.Drawing.Point(4, 22);
            this.tabCameraCal.Name = "tabCameraCal";
            this.tabCameraCal.Padding = new System.Windows.Forms.Padding(3);
            this.tabCameraCal.Size = new System.Drawing.Size(1079, 678);
            this.tabCameraCal.TabIndex = 5;
            this.tabCameraCal.Text = "Camera Cal";
            this.tabCameraCal.UseVisualStyleBackColor = true;
            // 
            // cameraMountCorrectionInput1
            // 
            this.cameraMountCorrectionInput1.DelXCorrCentiMeters = 0D;
            this.cameraMountCorrectionInput1.DelXCorrInches = 0D;
            this.cameraMountCorrectionInput1.DelXCorrMilliMeters = 0D;
            this.cameraMountCorrectionInput1.DelYCorrCentiMeters = 0D;
            this.cameraMountCorrectionInput1.DelYCorrInches = 0D;
            this.cameraMountCorrectionInput1.DelYCorrMilliMeters = 0D;
            this.cameraMountCorrectionInput1.Location = new System.Drawing.Point(474, 477);
            this.cameraMountCorrectionInput1.Name = "cameraMountCorrectionInput1";
            this.cameraMountCorrectionInput1.PitchCorrDeg = 0D;
            this.cameraMountCorrectionInput1.RollCorrDeg = 0D;
            this.cameraMountCorrectionInput1.Size = new System.Drawing.Size(358, 100);
            this.cameraMountCorrectionInput1.TabIndex = 4;
            this.cameraMountCorrectionInput1.YawCorrDeg = 0D;
            // 
            // cameraCalChessBdInput1
            // 
            this.cameraCalChessBdInput1.Location = new System.Drawing.Point(188, 477);
            this.cameraCalChessBdInput1.Name = "cameraCalChessBdInput1";
            this.cameraCalChessBdInput1.Size = new System.Drawing.Size(269, 106);
            this.cameraCalChessBdInput1.TabIndex = 3;
            // 
            // pnlCalImage
            // 
            this.pnlCalImage.AutoScroll = true;
            this.pnlCalImage.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.pnlCalImage.Controls.Add(this.pictureBoxCameraCalDisplay);
            this.pnlCalImage.Location = new System.Drawing.Point(188, 7);
            this.pnlCalImage.Name = "pnlCalImage";
            this.pnlCalImage.Size = new System.Drawing.Size(648, 467);
            this.pnlCalImage.TabIndex = 2;
            // 
            // pictureBoxCameraCalDisplay
            // 
            this.pictureBoxCameraCalDisplay.Location = new System.Drawing.Point(3, 3);
            this.pictureBoxCameraCalDisplay.Name = "pictureBoxCameraCalDisplay";
            this.pictureBoxCameraCalDisplay.Size = new System.Drawing.Size(640, 456);
            this.pictureBoxCameraCalDisplay.TabIndex = 0;
            this.pictureBoxCameraCalDisplay.TabStop = false;
            // 
            // cameraCalControl1
            // 
            this.cameraCalControl1.Location = new System.Drawing.Point(4, 7);
            this.cameraCalControl1.Name = "cameraCalControl1";
            this.cameraCalControl1.Size = new System.Drawing.Size(178, 528);
            this.cameraCalControl1.TabIndex = 0;
            // 
            // tabFeatureMatchProc
            // 
            this.tabFeatureMatchProc.Controls.Add(this.processTimerStatus1);
            this.tabFeatureMatchProc.Controls.Add(this.pictureBoxFeatureMatchProc);
            this.tabFeatureMatchProc.Controls.Add(this.featureMatchProcessControl1);
            this.tabFeatureMatchProc.Location = new System.Drawing.Point(4, 22);
            this.tabFeatureMatchProc.Name = "tabFeatureMatchProc";
            this.tabFeatureMatchProc.Padding = new System.Windows.Forms.Padding(3);
            this.tabFeatureMatchProc.Size = new System.Drawing.Size(1079, 678);
            this.tabFeatureMatchProc.TabIndex = 6;
            this.tabFeatureMatchProc.Text = "Feature Match";
            this.tabFeatureMatchProc.UseVisualStyleBackColor = true;
            // 
            // processTimerStatus1
            // 
            this.processTimerStatus1.Location = new System.Drawing.Point(185, 476);
            this.processTimerStatus1.Name = "processTimerStatus1";
            this.processTimerStatus1.Size = new System.Drawing.Size(197, 86);
            this.processTimerStatus1.TabIndex = 2;
            // 
            // pictureBoxFeatureMatchProc
            // 
            this.pictureBoxFeatureMatchProc.Location = new System.Drawing.Point(184, 6);
            this.pictureBoxFeatureMatchProc.Name = "pictureBoxFeatureMatchProc";
            this.pictureBoxFeatureMatchProc.Size = new System.Drawing.Size(655, 463);
            this.pictureBoxFeatureMatchProc.TabIndex = 1;
            this.pictureBoxFeatureMatchProc.TabStop = false;
            // 
            // featureMatchProcessControl1
            // 
            this.featureMatchProcessControl1.Location = new System.Drawing.Point(0, 3);
            this.featureMatchProcessControl1.Name = "featureMatchProcessControl1";
            this.featureMatchProcessControl1.Size = new System.Drawing.Size(178, 528);
            this.featureMatchProcessControl1.TabIndex = 0;
            // 
            // tabBrakeSetup
            // 
            this.tabBrakeSetup.Controls.Add(this.brakeLAPosCtrlSetupTab);
            this.tabBrakeSetup.Controls.Add(this.brakeLAConfigSetupTab);
            this.tabBrakeSetup.Location = new System.Drawing.Point(4, 22);
            this.tabBrakeSetup.Name = "tabBrakeSetup";
            this.tabBrakeSetup.Padding = new System.Windows.Forms.Padding(3);
            this.tabBrakeSetup.Size = new System.Drawing.Size(1079, 678);
            this.tabBrakeSetup.TabIndex = 9;
            this.tabBrakeSetup.Text = "Brake Setup";
            this.tabBrakeSetup.UseVisualStyleBackColor = true;
            // 
            // brakeLAPosCtrlSetupTab
            // 
            this.brakeLAPosCtrlSetupTab.Location = new System.Drawing.Point(8, 7);
            this.brakeLAPosCtrlSetupTab.Name = "brakeLAPosCtrlSetupTab";
            this.brakeLAPosCtrlSetupTab.Size = new System.Drawing.Size(349, 230);
            this.brakeLAPosCtrlSetupTab.TabIndex = 1;
            // 
            // brakeLAConfigSetupTab
            // 
            this.brakeLAConfigSetupTab.Location = new System.Drawing.Point(376, 7);
            this.brakeLAConfigSetupTab.Name = "brakeLAConfigSetupTab";
            this.brakeLAConfigSetupTab.Size = new System.Drawing.Size(595, 492);
            this.brakeLAConfigSetupTab.TabIndex = 0;
            // 
            // tabThrottleSetup
            // 
            this.tabThrottleSetup.Controls.Add(this.throttleLAConfigSetupTab);
            this.tabThrottleSetup.Controls.Add(this.throttleLAPosCtrlSetupTab);
            this.tabThrottleSetup.Location = new System.Drawing.Point(4, 22);
            this.tabThrottleSetup.Name = "tabThrottleSetup";
            this.tabThrottleSetup.Padding = new System.Windows.Forms.Padding(3);
            this.tabThrottleSetup.Size = new System.Drawing.Size(1079, 678);
            this.tabThrottleSetup.TabIndex = 10;
            this.tabThrottleSetup.Text = "Throttle Setup";
            this.tabThrottleSetup.UseVisualStyleBackColor = true;
            // 
            // throttleLAConfigSetupTab
            // 
            this.throttleLAConfigSetupTab.Location = new System.Drawing.Point(376, 5);
            this.throttleLAConfigSetupTab.Name = "throttleLAConfigSetupTab";
            this.throttleLAConfigSetupTab.Size = new System.Drawing.Size(591, 492);
            this.throttleLAConfigSetupTab.TabIndex = 3;
            // 
            // throttleLAPosCtrlSetupTab
            // 
            this.throttleLAPosCtrlSetupTab.Location = new System.Drawing.Point(6, 6);
            this.throttleLAPosCtrlSetupTab.Name = "throttleLAPosCtrlSetupTab";
            this.throttleLAPosCtrlSetupTab.Size = new System.Drawing.Size(349, 230);
            this.throttleLAPosCtrlSetupTab.TabIndex = 2;
            // 
            // tabPgTrackHead
            // 
            this.tabPgTrackHead.Controls.Add(this.headOrientationControlWidget1);
            this.tabPgTrackHead.Controls.Add(this.headOrientationCalWidget1);
            this.tabPgTrackHead.Controls.Add(this.pxBxTrackHeadDisplay);
            this.tabPgTrackHead.Controls.Add(this.headTrackingControlWidgt);
            this.tabPgTrackHead.Location = new System.Drawing.Point(4, 22);
            this.tabPgTrackHead.Name = "tabPgTrackHead";
            this.tabPgTrackHead.Padding = new System.Windows.Forms.Padding(3);
            this.tabPgTrackHead.Size = new System.Drawing.Size(1079, 678);
            this.tabPgTrackHead.TabIndex = 11;
            this.tabPgTrackHead.Text = "TrackHead";
            this.tabPgTrackHead.UseVisualStyleBackColor = true;
            // 
            // headOrientationControlWidget1
            // 
            this.headOrientationControlWidget1.Location = new System.Drawing.Point(260, 571);
            this.headOrientationControlWidget1.Name = "headOrientationControlWidget1";
            this.headOrientationControlWidget1.Size = new System.Drawing.Size(609, 107);
            this.headOrientationControlWidget1.TabIndex = 3;
            // 
            // headOrientationCalWidget1
            // 
            this.headOrientationCalWidget1.Location = new System.Drawing.Point(0, 507);
            this.headOrientationCalWidget1.Name = "headOrientationCalWidget1";
            this.headOrientationCalWidget1.Size = new System.Drawing.Size(260, 171);
            this.headOrientationCalWidget1.TabIndex = 2;
            // 
            // pxBxTrackHeadDisplay
            // 
            this.pxBxTrackHeadDisplay.Location = new System.Drawing.Point(260, 6);
            this.pxBxTrackHeadDisplay.Name = "pxBxTrackHeadDisplay";
            this.pxBxTrackHeadDisplay.Size = new System.Drawing.Size(778, 563);
            this.pxBxTrackHeadDisplay.TabIndex = 1;
            this.pxBxTrackHeadDisplay.TabStop = false;
            // 
            // headTrackingControlWidgt
            // 
            this.headTrackingControlWidgt.Location = new System.Drawing.Point(-1, 6);
            this.headTrackingControlWidgt.Name = "headTrackingControlWidgt";
            this.headTrackingControlWidgt.Size = new System.Drawing.Size(261, 482);
            this.headTrackingControlWidgt.TabIndex = 0;
            // 
            // btnConnectToVS
            // 
            this.btnConnectToVS.Location = new System.Drawing.Point(59, 69);
            this.btnConnectToVS.Name = "btnConnectToVS";
            this.btnConnectToVS.Size = new System.Drawing.Size(75, 23);
            this.btnConnectToVS.TabIndex = 2;
            this.btnConnectToVS.Text = "Connect";
            this.btnConnectToVS.UseVisualStyleBackColor = true;
            this.btnConnectToVS.Click += new System.EventHandler(this.btnConnectToVS_Click);
            // 
            // tbVSIPAddr
            // 
            this.tbVSIPAddr.Location = new System.Drawing.Point(12, 43);
            this.tbVSIPAddr.Name = "tbVSIPAddr";
            this.tbVSIPAddr.Size = new System.Drawing.Size(122, 20);
            this.tbVSIPAddr.TabIndex = 3;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 27);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(110, 13);
            this.label1.TabIndex = 4;
            this.label1.Text = "Vision System IP Addr";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(23, 27);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(67, 13);
            this.label2.TabIndex = 6;
            this.label2.Text = "Host IP Addr";
            this.label2.Visible = false;
            // 
            // tbHostIPAddr
            // 
            this.tbHostIPAddr.Location = new System.Drawing.Point(15, 43);
            this.tbHostIPAddr.Name = "tbHostIPAddr";
            this.tbHostIPAddr.Size = new System.Drawing.Size(75, 20);
            this.tbHostIPAddr.TabIndex = 5;
            this.tbHostIPAddr.Visible = false;
            // 
            // tbVSMsgInfo
            // 
            this.tbVSMsgInfo.Location = new System.Drawing.Point(12, 648);
            this.tbVSMsgInfo.Name = "tbVSMsgInfo";
            this.tbVSMsgInfo.Size = new System.Drawing.Size(137, 20);
            this.tbVSMsgInfo.TabIndex = 7;
            this.tbVSMsgInfo.Visible = false;
            // 
            // bnSSGPUProcessing
            // 
            this.bnSSGPUProcessing.Location = new System.Drawing.Point(15, 392);
            this.bnSSGPUProcessing.Name = "bnSSGPUProcessing";
            this.bnSSGPUProcessing.Size = new System.Drawing.Size(119, 23);
            this.bnSSGPUProcessing.TabIndex = 10;
            this.bnSSGPUProcessing.Text = "Enable GPU Proc";
            this.bnSSGPUProcessing.UseVisualStyleBackColor = true;
            this.bnSSGPUProcessing.Click += new System.EventHandler(this.bnSSGPUProcessing_Click);
            // 
            // btnSSVideoStream
            // 
            this.btnSSVideoStream.Location = new System.Drawing.Point(15, 421);
            this.btnSSVideoStream.Name = "btnSSVideoStream";
            this.btnSSVideoStream.Size = new System.Drawing.Size(119, 23);
            this.btnSSVideoStream.TabIndex = 11;
            this.btnSSVideoStream.Text = "Start Video Stream";
            this.btnSSVideoStream.UseVisualStyleBackColor = true;
            this.btnSSVideoStream.Click += new System.EventHandler(this.btnSSVideoStream_Click);
            // 
            // btnSSRecording
            // 
            this.btnSSRecording.Location = new System.Drawing.Point(15, 450);
            this.btnSSRecording.Name = "btnSSRecording";
            this.btnSSRecording.Size = new System.Drawing.Size(119, 23);
            this.btnSSRecording.TabIndex = 12;
            this.btnSSRecording.Text = "Start Recording";
            this.btnSSRecording.UseVisualStyleBackColor = true;
            this.btnSSRecording.Click += new System.EventHandler(this.btnSSRecording_Click);
            // 
            // btnShutdownVS
            // 
            this.btnShutdownVS.Location = new System.Drawing.Point(15, 666);
            this.btnShutdownVS.Name = "btnShutdownVS";
            this.btnShutdownVS.Size = new System.Drawing.Size(119, 23);
            this.btnShutdownVS.TabIndex = 13;
            this.btnShutdownVS.Text = "Shutdown V-System";
            this.btnShutdownVS.UseVisualStyleBackColor = true;
            this.btnShutdownVS.Click += new System.EventHandler(this.btnShutdownVS_Click);
            // 
            // btnGetVSInfo
            // 
            this.btnGetVSInfo.Enabled = false;
            this.btnGetVSInfo.Location = new System.Drawing.Point(15, 695);
            this.btnGetVSInfo.Name = "btnGetVSInfo";
            this.btnGetVSInfo.Size = new System.Drawing.Size(119, 23);
            this.btnGetVSInfo.TabIndex = 14;
            this.btnGetVSInfo.Text = "Get V-Sys Info";
            this.btnGetVSInfo.UseVisualStyleBackColor = true;
            this.btnGetVSInfo.Visible = false;
            this.btnGetVSInfo.Click += new System.EventHandler(this.btnGetVSInfo_Click);
            // 
            // timerMgrStatsUpdate
            // 
            this.timerMgrStatsUpdate.Interval = 1000;
            this.timerMgrStatsUpdate.Tick += new System.EventHandler(this.timerMgrStatsUpdate_Tick);
            // 
            // videreSystemStateControlForm
            // 
            this.videreSystemStateControlForm.Location = new System.Drawing.Point(12, 109);
            this.videreSystemStateControlForm.Name = "videreSystemStateControlForm";
            this.videreSystemStateControlForm.Size = new System.Drawing.Size(139, 248);
            this.videreSystemStateControlForm.TabIndex = 16;
            // 
            // imageCaptureControl1
            // 
            this.imageCaptureControl1.Location = new System.Drawing.Point(9, 492);
            this.imageCaptureControl1.Name = "imageCaptureControl1";
            this.imageCaptureControl1.Size = new System.Drawing.Size(134, 150);
            this.imageCaptureControl1.TabIndex = 15;
            // 
            // FalconVisionViewerMainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1254, 716);
            this.Controls.Add(this.videreSystemStateControlForm);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.tbVSIPAddr);
            this.Controls.Add(this.imageCaptureControl1);
            this.Controls.Add(this.btnGetVSInfo);
            this.Controls.Add(this.btnShutdownVS);
            this.Controls.Add(this.btnSSRecording);
            this.Controls.Add(this.btnSSVideoStream);
            this.Controls.Add(this.bnSSGPUProcessing);
            this.Controls.Add(this.tbVSMsgInfo);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.tbHostIPAddr);
            this.Controls.Add(this.btnConnectToVS);
            this.Controls.Add(this.tabMainScreen);
            this.Controls.Add(this.menuStrip1);
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "FalconVisionViewerMainForm";
            this.Text = "Falcon Vision Monitor and Viewer";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.FalconVisionViewerMainForm_FormClosing);
            this.tabMainScreen.ResumeLayout(false);
            this.tabCarControl.ResumeLayout(false);
            this.tabIMU.ResumeLayout(false);
            this.tabMgrStats.ResumeLayout(false);
            this.tabPgVehicleLocation.ResumeLayout(false);
            this.tabPgVehicleLocation.PerformLayout();
            this.tabPgSetup.ResumeLayout(false);
            this.tabMainPage.ResumeLayout(false);
            this.panelVideoImage1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxMainDsp)).EndInit();
            this.tabPgTargetInfo.ResumeLayout(false);
            this.tabCameraCal.ResumeLayout(false);
            this.pnlCalImage.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxCameraCalDisplay)).EndInit();
            this.tabFeatureMatchProc.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxFeatureMatchProc)).EndInit();
            this.tabBrakeSetup.ResumeLayout(false);
            this.tabThrottleSetup.ResumeLayout(false);
            this.tabPgTrackHead.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pxBxTrackHeadDisplay)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.TabControl tabMainScreen;
        private System.Windows.Forms.TabPage tabMainPage;
        private System.Windows.Forms.TabPage tabMgrStats;
        private System.Windows.Forms.Button btnConnectToVS;
        private System.Windows.Forms.TextBox tbVSIPAddr;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbHostIPAddr;
        private System.Windows.Forms.TextBox tbVSMsgInfo;
        private System.Windows.Forms.Button bnSSGPUProcessing;
        private System.Windows.Forms.Button btnSSVideoStream;
        private System.Windows.Forms.Button btnSSRecording;
        private System.Windows.Forms.Button btnShutdownVS;
        private System.Windows.Forms.Button btnGetVSInfo;
        private System.Windows.Forms.PictureBox pictureBoxMainDsp;
        private System.Windows.Forms.Panel panelVideoImage1;
        private GuiFormsWidgets.ManagerStatsUserControl MgrStatsCtrl_MgrNo_1;
        private GuiFormsWidgets.ManagerStatsUserControl MgrStatsCtrl_MgrNo_4;
        private GuiFormsWidgets.ManagerStatsUserControl MgrStatsCtrl_MgrNo_3;
        private GuiFormsWidgets.ManagerStatsUserControl MgrStatsCtrl_MgrNo_2;
        private System.Windows.Forms.Timer timerMgrStatsUpdate;
        private System.Windows.Forms.TabPage tabPgSetup;
        private GuiFormsWidgets.GeoCoordinateSystemSetup geoCoordinateSystemSetup1;
        private System.Windows.Forms.TabPage tabPgVehicleLocation;
        private GuiFormsWidgets.DisplayImageInfo DisplayImageInfoForm;
        private GuiFormsWidgets.UAVInertialStates UavInertialStatesFromImageInfo;
        private GuiFormsWidgets.ImageCaptureControl imageCaptureControl1;
        private GuiFormsWidgets.ImageCaptureSetupAndStatus imageCaptureSetupAndStatus1;
        private System.Windows.Forms.TabPage tabPgTargetInfo;
        private GuiFormsWidgets.VehicleAndImageLocation vehicleAndImageLocation1;
        private GuiFormsWidgets.ImageProcessControl imageProcessControl1;
        private System.Windows.Forms.TabPage tabCameraCal;
        private System.Windows.Forms.Panel pnlCalImage;
        private System.Windows.Forms.PictureBox pictureBoxCameraCalDisplay;
        private GuiFormsWidgets.CameraCalControl cameraCalControl1;
        private GuiFormsWidgets.CameraCalChessBdInput cameraCalChessBdInput1;
        private GuiFormsWidgets.CameraMountCorrectionInput cameraMountCorrectionInput1;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Button btnCopyInertialStates;
        private System.Windows.Forms.Label lbl_ISBottom2;
        private System.Windows.Forms.Label lbl_ISBottom1;
        private System.Windows.Forms.Label lbl_ISTop;
        private GuiFormsWidgets.UAVInertialStates SendInertialStatesToUAVCtrl;
        private GuiFormsWidgets.CameraOrientationControl cameraOrientationControl1;
        private GuiFormsWidgets.CameraParametersSetupWidget cameraParametersSetupWidget1;
        private GuiFormsWidgets.StreamRecordControlWidget streamRecordControlWidget1;
        private System.Windows.Forms.TabPage tabFeatureMatchProc;
        private GuiFormsWidgets.FeatureMatchProcessControl featureMatchProcessControl1;
        private System.Windows.Forms.PictureBox pictureBoxFeatureMatchProc;
        private GuiFormsWidgets.ProcessTimerStatus processTimerStatus1;
        private GuiFormsWidgets.BlobDetectorParameters blobDetectorParameters1;
        private System.Windows.Forms.Button btnGetListOfManagers;
        private System.Windows.Forms.TabPage tabCarControl;
        private CarCANBusMonitor.Widgets.LinearActuatorPositionControl BrakePositionControlWidget;
        private CarCANBusMonitor.Widgets.LinearActuatorPositionControl ThrottlePositionControlWidget;
        private CarCANBusMonitor.Widgets.SteeringControlWidget steeringControlWidget1;
        private GuiFormsWidgets.HeadOrientationSPWidget headOrientationSPMon;
        private System.Windows.Forms.TabPage tabIMU;
        private GuiFormsWidgets.HeadOrientationSPWidget headOrientationSPIMUTab;
        private GuiFormsWidgets.IMUControlWidget imuControlFrm;
        private System.Windows.Forms.TabPage tabBrakeSetup;
        private CarCANBusMonitor.Widgets.LinearActuatorPositionControl brakeLAPosCtrlSetupTab;
        private CarCANBusMonitor.Widgets.KarTechLinearActuatorSetupWidget brakeLAConfigSetupTab;
        private System.Windows.Forms.TabPage tabThrottleSetup;
        private CarCANBusMonitor.Widgets.KarTechLinearActuatorSetupWidget throttleLAConfigSetupTab;
        private CarCANBusMonitor.Widgets.LinearActuatorPositionControl throttleLAPosCtrlSetupTab;
        private System.Windows.Forms.TabPage tabPgTrackHead;
        private System.Windows.Forms.PictureBox pxBxTrackHeadDisplay;
        private GuiFormsWidgets.HeadTrackingControlWidget headTrackingControlWidgt;
        private GuiFormsWidgets.VidereSystemStateControlWidget videreSystemStateControlForm;
        private GuiFormsWidgets.HeadOrientationCalWidget headOrientationCalWidget1;
        private GuiFormsWidgets.HeadOrientationControlWidget headOrientationControlWidget1;
        private GuiFormsWidgets.VehicleControlParametersWidget vehicleControlParametersWidget1;
        private GuiFormsWidgets.GPSFixWidget gpsFixWidget1;
    }
}

