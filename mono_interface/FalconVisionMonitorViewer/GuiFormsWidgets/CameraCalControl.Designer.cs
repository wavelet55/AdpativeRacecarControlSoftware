namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class CameraCalControl
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.cbCalType = new System.Windows.Forms.ComboBox();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.tbMessageBox = new System.Windows.Forms.TextBox();
            this.btnRunCalibration = new System.Windows.Forms.Button();
            this.btnRejectImage = new System.Windows.Forms.Button();
            this.btnImageOK = new System.Windows.Forms.Button();
            this.btnCaptureImage = new System.Windows.Forms.Button();
            this.btnStartCal = new System.Windows.Forms.Button();
            this.btnResetCalProces = new System.Windows.Forms.Button();
            this.btnClearAllImages = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.tbNoCalImgages = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.tbCalDataBaseFilename = new System.Windows.Forms.TextBox();
            this.lblCalState = new System.Windows.Forms.Label();
            this.tbCalState = new System.Windows.Forms.TextBox();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.cbCalType);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.tbMessageBox);
            this.groupBox1.Controls.Add(this.btnRunCalibration);
            this.groupBox1.Controls.Add(this.btnRejectImage);
            this.groupBox1.Controls.Add(this.btnImageOK);
            this.groupBox1.Controls.Add(this.btnCaptureImage);
            this.groupBox1.Controls.Add(this.btnStartCal);
            this.groupBox1.Controls.Add(this.btnResetCalProces);
            this.groupBox1.Controls.Add(this.btnClearAllImages);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.tbNoCalImgages);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Controls.Add(this.tbCalDataBaseFilename);
            this.groupBox1.Controls.Add(this.lblCalState);
            this.groupBox1.Controls.Add(this.tbCalState);
            this.groupBox1.Location = new System.Drawing.Point(0, 3);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(175, 518);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Camera Calibration Control";
            // 
            // cbCalType
            // 
            this.cbCalType.FormattingEnabled = true;
            this.cbCalType.Location = new System.Drawing.Point(9, 33);
            this.cbCalType.Name = "cbCalType";
            this.cbCalType.Size = new System.Drawing.Size(153, 21);
            this.cbCalType.TabIndex = 16;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(6, 16);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(83, 13);
            this.label4.TabIndex = 15;
            this.label4.Text = "Calibration Type";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(6, 217);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(50, 13);
            this.label3.TabIndex = 14;
            this.label3.Text = "Message";
            // 
            // tbMessageBox
            // 
            this.tbMessageBox.Location = new System.Drawing.Point(6, 233);
            this.tbMessageBox.Multiline = true;
            this.tbMessageBox.Name = "tbMessageBox";
            this.tbMessageBox.ReadOnly = true;
            this.tbMessageBox.Size = new System.Drawing.Size(156, 100);
            this.tbMessageBox.TabIndex = 13;
            // 
            // btnRunCalibration
            // 
            this.btnRunCalibration.Location = new System.Drawing.Point(33, 480);
            this.btnRunCalibration.Name = "btnRunCalibration";
            this.btnRunCalibration.Size = new System.Drawing.Size(100, 23);
            this.btnRunCalibration.TabIndex = 12;
            this.btnRunCalibration.Text = "Run Calibration";
            this.btnRunCalibration.UseVisualStyleBackColor = true;
            this.btnRunCalibration.Click += new System.EventHandler(this.btnRunCalibration_Click);
            // 
            // btnRejectImage
            // 
            this.btnRejectImage.Location = new System.Drawing.Point(87, 441);
            this.btnRejectImage.Name = "btnRejectImage";
            this.btnRejectImage.Size = new System.Drawing.Size(75, 23);
            this.btnRejectImage.TabIndex = 11;
            this.btnRejectImage.Text = "Reject Img";
            this.btnRejectImage.UseVisualStyleBackColor = true;
            this.btnRejectImage.Click += new System.EventHandler(this.btnRejectImage_Click);
            // 
            // btnImageOK
            // 
            this.btnImageOK.Location = new System.Drawing.Point(6, 441);
            this.btnImageOK.Name = "btnImageOK";
            this.btnImageOK.Size = new System.Drawing.Size(75, 23);
            this.btnImageOK.TabIndex = 10;
            this.btnImageOK.Text = "Image Ok";
            this.btnImageOK.UseVisualStyleBackColor = true;
            this.btnImageOK.Click += new System.EventHandler(this.btnImageOK_Click);
            // 
            // btnCaptureImage
            // 
            this.btnCaptureImage.Location = new System.Drawing.Point(6, 412);
            this.btnCaptureImage.Name = "btnCaptureImage";
            this.btnCaptureImage.Size = new System.Drawing.Size(75, 23);
            this.btnCaptureImage.TabIndex = 9;
            this.btnCaptureImage.Text = "Capture Img";
            this.btnCaptureImage.UseVisualStyleBackColor = true;
            this.btnCaptureImage.Click += new System.EventHandler(this.btnCaptureImage_Click);
            // 
            // btnStartCal
            // 
            this.btnStartCal.Location = new System.Drawing.Point(6, 383);
            this.btnStartCal.Name = "btnStartCal";
            this.btnStartCal.Size = new System.Drawing.Size(75, 23);
            this.btnStartCal.TabIndex = 8;
            this.btnStartCal.Text = "Start Cal";
            this.btnStartCal.UseVisualStyleBackColor = true;
            this.btnStartCal.Click += new System.EventHandler(this.btnStartCal_Click);
            // 
            // btnResetCalProces
            // 
            this.btnResetCalProces.Location = new System.Drawing.Point(6, 354);
            this.btnResetCalProces.Name = "btnResetCalProces";
            this.btnResetCalProces.Size = new System.Drawing.Size(75, 23);
            this.btnResetCalProces.TabIndex = 7;
            this.btnResetCalProces.Text = "Reset";
            this.btnResetCalProces.UseVisualStyleBackColor = true;
            this.btnResetCalProces.Click += new System.EventHandler(this.btnResetCalProces_Click);
            // 
            // btnClearAllImages
            // 
            this.btnClearAllImages.Location = new System.Drawing.Point(71, 138);
            this.btnClearAllImages.Name = "btnClearAllImages";
            this.btnClearAllImages.Size = new System.Drawing.Size(91, 23);
            this.btnClearAllImages.TabIndex = 6;
            this.btnClearAllImages.Text = "Clear Images";
            this.btnClearAllImages.UseVisualStyleBackColor = true;
            this.btnClearAllImages.Click += new System.EventHandler(this.btnClearAllImages_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(6, 115);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(87, 13);
            this.label2.TabIndex = 5;
            this.label2.Text = "Num Cal Images:";
            // 
            // tbNoCalImgages
            // 
            this.tbNoCalImgages.Location = new System.Drawing.Point(112, 112);
            this.tbNoCalImgages.Name = "tbNoCalImgages";
            this.tbNoCalImgages.ReadOnly = true;
            this.tbNoCalImgages.Size = new System.Drawing.Size(50, 20);
            this.tbNoCalImgages.TabIndex = 4;
            this.tbNoCalImgages.Text = "0";
            this.tbNoCalImgages.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(6, 64);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(94, 13);
            this.label1.TabIndex = 3;
            this.label1.Text = "Cal Base Filename";
            // 
            // tbCalDataBaseFilename
            // 
            this.tbCalDataBaseFilename.Location = new System.Drawing.Point(6, 77);
            this.tbCalDataBaseFilename.Name = "tbCalDataBaseFilename";
            this.tbCalDataBaseFilename.Size = new System.Drawing.Size(156, 20);
            this.tbCalDataBaseFilename.TabIndex = 2;
            this.tbCalDataBaseFilename.Text = "CameraCalData";
            // 
            // lblCalState
            // 
            this.lblCalState.AutoSize = true;
            this.lblCalState.Location = new System.Drawing.Point(6, 181);
            this.lblCalState.Name = "lblCalState";
            this.lblCalState.Size = new System.Drawing.Size(50, 13);
            this.lblCalState.TabIndex = 1;
            this.lblCalState.Text = "Cal State";
            // 
            // tbCalState
            // 
            this.tbCalState.Location = new System.Drawing.Point(6, 194);
            this.tbCalState.Name = "tbCalState";
            this.tbCalState.ReadOnly = true;
            this.tbCalState.Size = new System.Drawing.Size(156, 20);
            this.tbCalState.TabIndex = 0;
            // 
            // CameraCalControl
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.groupBox1);
            this.Name = "CameraCalControl";
            this.Size = new System.Drawing.Size(178, 528);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbMessageBox;
        private System.Windows.Forms.Button btnRunCalibration;
        private System.Windows.Forms.Button btnRejectImage;
        private System.Windows.Forms.Button btnImageOK;
        private System.Windows.Forms.Button btnCaptureImage;
        private System.Windows.Forms.Button btnStartCal;
        private System.Windows.Forms.Button btnResetCalProces;
        private System.Windows.Forms.Button btnClearAllImages;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbNoCalImgages;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox tbCalDataBaseFilename;
        private System.Windows.Forms.Label lblCalState;
        private System.Windows.Forms.TextBox tbCalState;
        private System.Windows.Forms.ComboBox cbCalType;
        private System.Windows.Forms.Label label4;
    }
}
