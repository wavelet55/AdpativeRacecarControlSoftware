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
    public partial class CameraCalControl : UserControl
    {
        public VisionCmdProcess VisionCmdProc;

        private CameraCalControlMsg CameraCalCtrlMsg;

        private bool ImageOk = false;

        public CameraCalChessBdInput CameraCalChessBdInp = null;

        public CameraMountCorrectionInput CameraMountCorrInp = null;

        public CameraCalControl()
        {
            InitializeComponent();

            CameraCalCtrlMsg = new CameraCalControlMsg();

            cbCalType.Items.Add(CameraCalibrationType_e.CCT_2DPlaneCheckerBoard);
            cbCalType.SelectedIndex = 0;
        }

        private void SetCommonCameraCalMsgPrams()
        {
            CameraCalCtrlMsg.NumberOfRows  = CameraCalChessBdInp.NumberOfRows;
            CameraCalCtrlMsg.NumberOfCols  = CameraCalChessBdInp.NumberOfCols;
            CameraCalCtrlMsg.SquareSizeMilliMeters = CameraCalChessBdInp.SquareSizeMilliMeters;

            if (CameraMountCorrInp != null)
            {
                CameraMountCorrInp.ReadValues();
                CameraCalCtrlMsg.YawCorrectionDegrees = CameraMountCorrInp.YawCorrDeg;
                CameraCalCtrlMsg.PitchCorrectionDegrees = CameraMountCorrInp.PitchCorrDeg;
                CameraCalCtrlMsg.RollCorrectionDegrees = CameraMountCorrInp.RollCorrDeg;
                CameraCalCtrlMsg.DelXCorrectionCentiMeters = CameraMountCorrInp.DelXCorrCentiMeters;
                CameraCalCtrlMsg.DelYCorrectionCentiMeters = CameraMountCorrInp.DelYCorrCentiMeters;
            }

            CameraCalCtrlMsg.CameraCalBaseFilename = tbCalDataBaseFilename.Text;
            CameraCalCtrlMsg.CameraCalibrationType = (CameraCalibrationType_e)cbCalType.SelectedIndex;
        }

        private void btnClearAllImages_Click(object sender, EventArgs e)
        {
            SetCommonCameraCalMsgPrams();
            CameraCalCtrlMsg.CameraCalCmd = CameraCalControlMsg.CameraCalCmd_e.ClearImageSet;
            VisionCmdProc.SendCameraCalControlMsg(CameraCalCtrlMsg);
        }

        private void btnResetCalProces_Click(object sender, EventArgs e)
        {
            if (CameraCalChessBdInp != null)
                CameraCalChessBdInp.ReadParameters();

            SetCommonCameraCalMsgPrams();
            CameraCalCtrlMsg.CameraCalCmd = CameraCalControlMsg.CameraCalCmd_e.Reset;
            VisionCmdProc.SendCameraCalControlMsg(CameraCalCtrlMsg);
        }

        private void btnStartCal_Click(object sender, EventArgs e)
        {
            if (CameraCalChessBdInp != null && !CameraCalChessBdInp.ReadParameters())
            {
                SetCommonCameraCalMsgPrams();
                CameraCalCtrlMsg.CameraCalCmd = CameraCalControlMsg.CameraCalCmd_e.StreamImages;
                VisionCmdProc.SendCameraCalControlMsg(CameraCalCtrlMsg);
            }
            else
            {
                tbMessageBox.Text = "Error Reading Chess Bd. Parameters";
            }
        }

        private void btnCaptureImage_Click(object sender, EventArgs e)
        {
            SetCommonCameraCalMsgPrams();
            CameraCalCtrlMsg.CameraCalCmd = CameraCalControlMsg.CameraCalCmd_e.CaptureImage;
            VisionCmdProc.SendCameraCalControlMsg(CameraCalCtrlMsg);
        }

        private void btnImageOK_Click(object sender, EventArgs e)
        {
            SetCommonCameraCalMsgPrams();
            CameraCalCtrlMsg.CameraCalCmd = CameraCalControlMsg.CameraCalCmd_e.SetImageOk;
            VisionCmdProc.SendCameraCalControlMsg(CameraCalCtrlMsg);
        }

        private void btnRejectImage_Click(object sender, EventArgs e)
        {
            SetCommonCameraCalMsgPrams();
            CameraCalCtrlMsg.CameraCalCmd = CameraCalControlMsg.CameraCalCmd_e.RejectImage;
            VisionCmdProc.SendCameraCalControlMsg(CameraCalCtrlMsg);
        }

        private void btnRunCalibration_Click(object sender, EventArgs e)
        {
            SetCommonCameraCalMsgPrams();
            CameraCalCtrlMsg.CameraCalCmd = CameraCalControlMsg.CameraCalCmd_e.RunCalProcess;
            VisionCmdProc.SendCameraCalControlMsg(CameraCalCtrlMsg);
        }

        public void processCameraCalStatusMessage(CameraCalStatusMsg ccsMsg)
        {
            tbCalState.Text = ccsMsg.CameraCalState.ToString();
            tbNoCalImgages.Text = ccsMsg.NumberOfImagesCaptured.ToString();
            tbMessageBox.Text = ccsMsg.CameraCalMsg;
            ImageOk = ccsMsg.ImageOk;
            switch (ccsMsg.CameraCalState)
            {
                case CameraCalStatusMsg.CameraCalState_e.Reset:
                    btnImageOK.BackColor = Color.LightGray;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnCaptureImage.BackColor = Color.LightGray;
                    btnRunCalibration.BackColor = Color.LightGray;
                    btnStartCal.BackColor = Color.LightGray;
                    break;
                case CameraCalStatusMsg.CameraCalState_e.WaitForStart:
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
                case CameraCalStatusMsg.CameraCalState_e.StreamImages:
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
                case CameraCalStatusMsg.CameraCalState_e.ImageValidate:
                    btnCaptureImage.BackColor = Color.LightGray;
                    break;
                case CameraCalStatusMsg.CameraCalState_e.ImageCapturedWait:
                    btnCaptureImage.BackColor = Color.LightGray;
                    if (ccsMsg.ImageOk)
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
                case CameraCalStatusMsg.CameraCalState_e.CalProcess:
                    btnImageOK.BackColor = Color.LightGray;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnStartCal.BackColor = Color.LightGray;
                    btnCaptureImage.BackColor = Color.LightGray;

                    break;
                case CameraCalStatusMsg.CameraCalState_e.CalComplete:
                    btnImageOK.BackColor = Color.LightGray;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnStartCal.BackColor = Color.LightGray;
                    btnCaptureImage.BackColor = Color.LightGray;
                    btnRunCalibration.BackColor = Color.Blue;
                    break;
                case CameraCalStatusMsg.CameraCalState_e.CalError:
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
