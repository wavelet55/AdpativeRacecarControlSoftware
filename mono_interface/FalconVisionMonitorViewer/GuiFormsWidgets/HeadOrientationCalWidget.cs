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
    public partial class HeadOrientationCalWidget : UserControl
    {

        public VisionCmdProcess VisionCmdProc;

        private CameraCalControlMsg CameraCalCtrlMsg;

        private bool ImageOk = false;

        enum CalRunState_e
        {
            Reset,
            Running,
            Complete
        }

        CalRunState_e CalRunState = CalRunState_e.Reset;

        public HeadOrientationCalWidget()
        {
            InitializeComponent();
            CalRunState = CalRunState_e.Reset;
            CameraCalCtrlMsg = new CameraCalControlMsg();
        }

        private void btnCalStart_Click(object sender, EventArgs e)
        {
            if (CalRunState == CalRunState_e.Reset)
            {
                CameraCalCtrlMsg.CameraCalCmd = CameraCalControlMsg.CameraCalCmd_e.Reset;
                VisionCmdProc.SendCameraCalControlMsg(CameraCalCtrlMsg);
                btnCalStart.Text = "Reset";
                btnCalStart.BackColor = System.Drawing.Color.Yellow;
                CalRunState = CalRunState_e.Running;
            }
            else if (CalRunState == CalRunState_e.Running)
            {
                CameraCalCtrlMsg.CameraCalCmd = CameraCalControlMsg.CameraCalCmd_e.Reset;
                VisionCmdProc.SendCameraCalControlMsg(CameraCalCtrlMsg);
                btnCalStart.Text = "Start";
                btnCalStart.BackColor = System.Drawing.Color.Gray;
                CalRunState = CalRunState_e.Reset;
            }
            else
            {
                CameraCalCtrlMsg.CameraCalCmd = CameraCalControlMsg.CameraCalCmd_e.Reset;
                VisionCmdProc.SendCameraCalControlMsg(CameraCalCtrlMsg);
                btnCalStart.Text = "ReStart";
                btnCalStart.BackColor = System.Drawing.Color.Yellow;
                CalRunState = CalRunState_e.Reset;
            }
        }

        private void btnCaptureImage_Click(object sender, EventArgs e)
        {
            CameraCalCtrlMsg.CameraCalCmd = CameraCalControlMsg.CameraCalCmd_e.CaptureImage;
            VisionCmdProc.SendCameraCalControlMsg(CameraCalCtrlMsg);
        }

        private void btnImageOK_Click(object sender, EventArgs e)
        {
            CameraCalCtrlMsg.CameraCalCmd = CameraCalControlMsg.CameraCalCmd_e.SetImageOk;
            VisionCmdProc.SendCameraCalControlMsg(CameraCalCtrlMsg);
        }

        private void btnRejectImage_Click(object sender, EventArgs e)
        {
            CameraCalCtrlMsg.CameraCalCmd = CameraCalControlMsg.CameraCalCmd_e.RejectImage;
            VisionCmdProc.SendCameraCalControlMsg(CameraCalCtrlMsg);
        }


        public void processCameraCalStatusMessage(CameraCalStatusMsg ccsMsg)
        {
            tbCalState.Text = ccsMsg.CameraCalState.ToString();
            tbImageNumber.Text = ccsMsg.NumberOfImagesCaptured.ToString();
            tbCalMessage.Text = ccsMsg.CameraCalMsg;
            ImageOk = ccsMsg.ImageOk;
            switch (ccsMsg.CameraCalState)
            {
                case CameraCalStatusMsg.CameraCalState_e.Reset:
                    btnImageOK.BackColor = Color.LightGray;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnCaptureImage.BackColor = Color.LightGray;
                    btnCalStart.BackColor = Color.LightGray;
                    break;
                case CameraCalStatusMsg.CameraCalState_e.WaitForStart:
                    btnImageOK.BackColor = Color.LightGray;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnCaptureImage.BackColor = Color.LightGray;
                    btnCalStart.BackColor = Color.Yellow;
                    break;
                case CameraCalStatusMsg.CameraCalState_e.StreamImages:
                    btnImageOK.BackColor = Color.LightGray;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnCaptureImage.BackColor = Color.Green;
                    btnCalStart.BackColor = Color.Yellow;
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
                    btnCalStart.BackColor = Color.LightGreen;
                    btnCaptureImage.BackColor = Color.LightGray;

                    break;
                case CameraCalStatusMsg.CameraCalState_e.CalComplete:
                    btnImageOK.BackColor = Color.LightGray;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnCalStart.BackColor = Color.LightGreen;
                    btnCalStart.Text = "Cal Done";
                    CalRunState = CalRunState_e.Complete;
                    btnCaptureImage.BackColor = Color.LightGray;
                    break;
                case CameraCalStatusMsg.CameraCalState_e.CalError:
                    btnImageOK.BackColor = Color.LightGray;
                    btnCalStart.BackColor = Color.Red;
                    btnRejectImage.BackColor = Color.LightGray;
                    btnCaptureImage.BackColor = Color.LightGray;
                    break;
            }
        }


    }
}
