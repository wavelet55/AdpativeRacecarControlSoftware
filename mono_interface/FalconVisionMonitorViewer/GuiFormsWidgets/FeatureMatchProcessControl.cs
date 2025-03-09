using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using VisionBridge.Messages;


namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    public partial class FeatureMatchProcessControl : UserControl
    {
        public VisionCmdProcess VisionCmdProc;

        private FeatureMatchProcCtrlPBMsg FeatureMatchCtrlMsg;

        private bool ImageOk = false;

        public CameraCalChessBdInput CameraCalChessBdInp = null;

        public CameraMountCorrectionInput CameraMountCorrInp = null;

        public FeatureMatchProcessControl()
        {
            InitializeComponent();

            FeatureMatchCtrlMsg = new FeatureMatchProcCtrlPBMsg();

            cbFeatureExtractionType.Items.Add(FeatureExtractionTypeRoutine_e.ORB);
            cbFeatureExtractionType.SelectedIndex = 0;

            cbFeatureMatchType.Items.Add(FeatureMatchTypeRoutine_e.BruteForce);
            cbFeatureMatchType.Items.Add(FeatureMatchTypeRoutine_e.FLANN);
            cbFeatureMatchType.SelectedIndex = 0;

            cbImagePostProcessMethod.Items.Add(FMImagePostProcessMethod_e.None);
            cbImagePostProcessMethod.Items.Add(FMImagePostProcessMethod_e.MarkObjectFoundCircle);
            cbImagePostProcessMethod.Items.Add(FMImagePostProcessMethod_e.MarkObjectFoundRect);
            cbImagePostProcessMethod.Items.Add(FMImagePostProcessMethod_e.GenFeatureMap);
            cbImagePostProcessMethod.Items.Add(FMImagePostProcessMethod_e.GenFeatureMapAndMarkObjCircle);
            cbImagePostProcessMethod.Items.Add(FMImagePostProcessMethod_e.GenFeatureMapAndMarkObjRect);
            cbImagePostProcessMethod.SelectedIndex = 0;
        }

        private void SetCommonFeatureMatchMsgPrams()
        {
            //FeatureMatchCtrlMsg.NumberOfRows  = CameraCalChessBdInp.NumberOfRows;
            //FeatureMatchCtrlMsg.NumberOfCols  = CameraCalChessBdInp.NumberOfCols;
            //FeatureMatchCtrlMsg.SquareSizeMilliMeters = CameraCalChessBdInp.SquareSizeMilliMeters;

            if (CameraMountCorrInp != null)
            {
                CameraMountCorrInp.ReadValues();
            }

            FeatureMatchCtrlMsg.FeatureExtractionTypeRoutine = (FeatureExtractionTypeRoutine_e)cbFeatureExtractionType.SelectedIndex;
            FeatureMatchCtrlMsg.FeatureMatchTypeRoutine = (FeatureMatchTypeRoutine_e)cbFeatureMatchType.SelectedIndex;
            FeatureMatchCtrlMsg.FMImagePostProcessMethod = (FMImagePostProcessMethod_e)cbImagePostProcessMethod.SelectedIndex;
        }

        private void btnClearAllImages_Click(object sender, EventArgs e)
        {
            SetCommonFeatureMatchMsgPrams();
            FeatureMatchCtrlMsg.FeatureMatchingProcCmd = FeatureMatchingProcCmd_e.ClearImageSet;
            VisionCmdProc.SendFeatureMatchProcCtrlMsg(FeatureMatchCtrlMsg);
        }

        private void btnResetCalProces_Click(object sender, EventArgs e)
        {
            if (CameraCalChessBdInp != null)
                CameraCalChessBdInp.ReadParameters();

            SetCommonFeatureMatchMsgPrams();
            FeatureMatchCtrlMsg.FeatureMatchingProcCmd = FeatureMatchingProcCmd_e.Reset;
            VisionCmdProc.SendFeatureMatchProcCtrlMsg(FeatureMatchCtrlMsg);
        }

        private void btnStartCal_Click(object sender, EventArgs e)
        {
            SetCommonFeatureMatchMsgPrams();
            FeatureMatchCtrlMsg.FeatureMatchingProcCmd = FeatureMatchingProcCmd_e.StreamImages;
            VisionCmdProc.SendFeatureMatchProcCtrlMsg(FeatureMatchCtrlMsg);
        }

        private void btnCaptureImage_Click(object sender, EventArgs e)
        {
            SetCommonFeatureMatchMsgPrams();
            FeatureMatchCtrlMsg.FeatureMatchingProcCmd = FeatureMatchingProcCmd_e.CaptureImage;
            VisionCmdProc.SendFeatureMatchProcCtrlMsg(FeatureMatchCtrlMsg);
        }

        private void btnImageOK_Click(object sender, EventArgs e)
        {
            SetCommonFeatureMatchMsgPrams();
            FeatureMatchCtrlMsg.FeatureMatchingProcCmd = FeatureMatchingProcCmd_e.SetImageOk;
            VisionCmdProc.SendFeatureMatchProcCtrlMsg(FeatureMatchCtrlMsg);
        }

        private void btnRejectImage_Click(object sender, EventArgs e)
        {
            SetCommonFeatureMatchMsgPrams();
            FeatureMatchCtrlMsg.FeatureMatchingProcCmd = FeatureMatchingProcCmd_e.RejectImage;
            VisionCmdProc.SendFeatureMatchProcCtrlMsg(FeatureMatchCtrlMsg);
        }

        private void btnRunCalibration_Click(object sender, EventArgs e)
        {
            SetCommonFeatureMatchMsgPrams();
            FeatureMatchCtrlMsg.FeatureMatchingProcCmd = FeatureMatchingProcCmd_e.RunImageProcess;
            VisionCmdProc.SendFeatureMatchProcCtrlMsg(FeatureMatchCtrlMsg);
        }

        public void processFeatureMatchlStatusMessage(FeatureMatchProcStatusPBMsg ccsMsg)
        {
            tbCalState.Text = ccsMsg.FeatureMatchingState.ToString();
            tbNoCalImgages.Text = ccsMsg.NumberOfImagesCaptured.ToString();
            tbMessageBox.Text = ccsMsg.StatusMessage;
            //ImageOk = ccsMsg.;
            switch (ccsMsg.FeatureMatchingState)
            {
                case FeatureMatchingState_e.Reset:
                    btnImageOK.BackColor = Color.LightGray;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnCaptureImage.BackColor = Color.LightGray;
                    btnRunCalibration.BackColor = Color.LightGray;
                    btnStartCal.BackColor = Color.LightGray;
                    break;
                case FeatureMatchingState_e.WaitForStart:
                    btnImageOK.BackColor = Color.LightGray;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnCaptureImage.BackColor = Color.LightGray;
                    btnStartCal.BackColor = Color.Green;
                    if (ccsMsg.NumberOfImagesCaptured > 3)
                    {
                        btnRunCalibration.BackColor = Color.LightGreen;
                    }
                    else
                    {
                        btnRunCalibration.BackColor = Color.LightGray;
                    }
                    break;
                case FeatureMatchingState_e.StreamImages:
                    btnImageOK.BackColor = Color.LightGray;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnCaptureImage.BackColor = Color.Green;
                    btnStartCal.BackColor = Color.LightGray;
                    if (ccsMsg.NumberOfImagesCaptured > 3)
                    {
                        btnRunCalibration.BackColor = Color.LightGreen;
                    }
                    else
                    {
                        btnRunCalibration.BackColor = Color.LightGray;
                    }
                    break;
                case FeatureMatchingState_e.ImageValidate:
                    btnCaptureImage.BackColor = Color.LightGray;
                    break;
                case FeatureMatchingState_e.ImageCapturedWait:
                    btnCaptureImage.BackColor = Color.LightGray;
                    if (ccsMsg.StatusValI_1 != 0)
                    {
                        btnImageOK.BackColor = Color.LightGreen;
                        btnRejectImage.BackColor = Color.LightYellow;
                    }
                    else
                    {
                        btnImageOK.BackColor = Color.LightYellow;
                        btnRejectImage.BackColor = Color.LightGreen;
                    }
                    break;
                case FeatureMatchingState_e.FMProcess:
                    btnImageOK.BackColor = Color.LightGray;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnStartCal.BackColor = Color.LightGray;
                    btnCaptureImage.BackColor = Color.LightGray;

                    break;
                case FeatureMatchingState_e.FMComplete:
                    btnImageOK.BackColor = Color.LightGray;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnStartCal.BackColor = Color.LightGray;
                    btnCaptureImage.BackColor = Color.LightGray;
                    btnRunCalibration.BackColor = Color.Blue;
                    break;
                case FeatureMatchingState_e.FMError:
                    btnImageOK.BackColor = Color.LightGray;
                    btnStartCal.BackColor = Color.LightGray;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnCaptureImage.BackColor = Color.LightGray;
                    btnRunCalibration.BackColor = Color.Red;
                    break;
            }
        }

    }
}
