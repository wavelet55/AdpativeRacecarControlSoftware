namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class ImageProcessControl
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
            this.gbImageProcessControl = new System.Windows.Forms.GroupBox();
            this.label3 = new System.Windows.Forms.Label();
            this.btnImgProcEnabled = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.tbGPUStatus = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.tbActualTgtDetectType = new System.Windows.Forms.TextBox();
            this.btnSendImgProcCtrlMsg = new System.Windows.Forms.Button();
            this.chkBxGPUTgtDetectionEnabled = new System.Windows.Forms.CheckBox();
            this.lbTgtDetectionType = new System.Windows.Forms.Label();
            this.cbTgtDetectionType = new System.Windows.Forms.ComboBox();
            this.cbVisionProcessorType = new System.Windows.Forms.ComboBox();
            this.label4 = new System.Windows.Forms.Label();
            this.tbVisionProcModeSelected = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.gbImageProcessControl.SuspendLayout();
            this.SuspendLayout();
            // 
            // gbImageProcessControl
            // 
            this.gbImageProcessControl.Controls.Add(this.label5);
            this.gbImageProcessControl.Controls.Add(this.tbVisionProcModeSelected);
            this.gbImageProcessControl.Controls.Add(this.label4);
            this.gbImageProcessControl.Controls.Add(this.cbVisionProcessorType);
            this.gbImageProcessControl.Controls.Add(this.label3);
            this.gbImageProcessControl.Controls.Add(this.btnImgProcEnabled);
            this.gbImageProcessControl.Controls.Add(this.label2);
            this.gbImageProcessControl.Controls.Add(this.tbGPUStatus);
            this.gbImageProcessControl.Controls.Add(this.label1);
            this.gbImageProcessControl.Controls.Add(this.tbActualTgtDetectType);
            this.gbImageProcessControl.Controls.Add(this.btnSendImgProcCtrlMsg);
            this.gbImageProcessControl.Controls.Add(this.chkBxGPUTgtDetectionEnabled);
            this.gbImageProcessControl.Controls.Add(this.lbTgtDetectionType);
            this.gbImageProcessControl.Controls.Add(this.cbTgtDetectionType);
            this.gbImageProcessControl.Location = new System.Drawing.Point(4, 4);
            this.gbImageProcessControl.Name = "gbImageProcessControl";
            this.gbImageProcessControl.Size = new System.Drawing.Size(329, 209);
            this.gbImageProcessControl.TabIndex = 0;
            this.gbImageProcessControl.TabStop = false;
            this.gbImageProcessControl.Text = "Image Process Control";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(260, 76);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(37, 13);
            this.label3.TabIndex = 9;
            this.label3.Text = "Status";
            // 
            // btnImgProcEnabled
            // 
            this.btnImgProcEnabled.Location = new System.Drawing.Point(247, 90);
            this.btnImgProcEnabled.Name = "btnImgProcEnabled";
            this.btnImgProcEnabled.Size = new System.Drawing.Size(72, 23);
            this.btnImgProcEnabled.TabIndex = 8;
            this.btnImgProcEnabled.Text = "Disabled";
            this.btnImgProcEnabled.UseVisualStyleBackColor = true;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(173, 76);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(30, 13);
            this.label2.TabIndex = 7;
            this.label2.Text = "GPU";
            // 
            // tbGPUStatus
            // 
            this.tbGPUStatus.Location = new System.Drawing.Point(173, 92);
            this.tbGPUStatus.Name = "tbGPUStatus";
            this.tbGPUStatus.Size = new System.Drawing.Size(55, 20);
            this.tbGPUStatus.TabIndex = 6;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(17, 76);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(147, 13);
            this.label1.TabIndex = 5;
            this.label1.Text = "Actual Target Detection Type";
            // 
            // tbActualTgtDetectType
            // 
            this.tbActualTgtDetectType.Location = new System.Drawing.Point(17, 92);
            this.tbActualTgtDetectType.Name = "tbActualTgtDetectType";
            this.tbActualTgtDetectType.Size = new System.Drawing.Size(147, 20);
            this.tbActualTgtDetectType.TabIndex = 4;
            // 
            // btnSendImgProcCtrlMsg
            // 
            this.btnSendImgProcCtrlMsg.Location = new System.Drawing.Point(244, 170);
            this.btnSendImgProcCtrlMsg.Name = "btnSendImgProcCtrlMsg";
            this.btnSendImgProcCtrlMsg.Size = new System.Drawing.Size(75, 23);
            this.btnSendImgProcCtrlMsg.TabIndex = 3;
            this.btnSendImgProcCtrlMsg.Text = "Send";
            this.btnSendImgProcCtrlMsg.UseVisualStyleBackColor = true;
            this.btnSendImgProcCtrlMsg.Click += new System.EventHandler(this.btnSendImgProcCtrlMsg_Click);
            // 
            // chkBxGPUTgtDetectionEnabled
            // 
            this.chkBxGPUTgtDetectionEnabled.AutoSize = true;
            this.chkBxGPUTgtDetectionEnabled.Location = new System.Drawing.Point(173, 147);
            this.chkBxGPUTgtDetectionEnabled.Name = "chkBxGPUTgtDetectionEnabled";
            this.chkBxGPUTgtDetectionEnabled.Size = new System.Drawing.Size(133, 17);
            this.chkBxGPUTgtDetectionEnabled.TabIndex = 2;
            this.chkBxGPUTgtDetectionEnabled.Text = "Use GPU Acceleration";
            this.chkBxGPUTgtDetectionEnabled.UseVisualStyleBackColor = true;
            // 
            // lbTgtDetectionType
            // 
            this.lbTgtDetectionType.AutoSize = true;
            this.lbTgtDetectionType.Location = new System.Drawing.Point(17, 127);
            this.lbTgtDetectionType.Name = "lbTgtDetectionType";
            this.lbTgtDetectionType.Size = new System.Drawing.Size(114, 13);
            this.lbTgtDetectionType.TabIndex = 1;
            this.lbTgtDetectionType.Text = "Target Detection Type";
            // 
            // cbTgtDetectionType
            // 
            this.cbTgtDetectionType.FormattingEnabled = true;
            this.cbTgtDetectionType.Location = new System.Drawing.Point(17, 143);
            this.cbTgtDetectionType.Name = "cbTgtDetectionType";
            this.cbTgtDetectionType.Size = new System.Drawing.Size(147, 21);
            this.cbTgtDetectionType.TabIndex = 0;
            this.cbTgtDetectionType.SelectedIndexChanged += new System.EventHandler(this.cbTgtDetectionType_SelectedIndexChanged);
            // 
            // cbVisionProcessorType
            // 
            this.cbVisionProcessorType.FormattingEnabled = true;
            this.cbVisionProcessorType.Location = new System.Drawing.Point(17, 34);
            this.cbVisionProcessorType.Name = "cbVisionProcessorType";
            this.cbVisionProcessorType.Size = new System.Drawing.Size(147, 21);
            this.cbVisionProcessorType.TabIndex = 10;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(17, 18);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(85, 13);
            this.label4.TabIndex = 11;
            this.label4.Text = "Vision Processor";
            // 
            // tbVisionProcModeSelected
            // 
            this.tbVisionProcModeSelected.Location = new System.Drawing.Point(176, 35);
            this.tbVisionProcModeSelected.Name = "tbVisionProcModeSelected";
            this.tbVisionProcModeSelected.Size = new System.Drawing.Size(144, 20);
            this.tbVisionProcModeSelected.TabIndex = 12;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(176, 19);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(130, 13);
            this.label5.TabIndex = 13;
            this.label5.Text = "Selected Vision Processor";
            // 
            // ImageProcessControl
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.gbImageProcessControl);
            this.Name = "ImageProcessControl";
            this.Size = new System.Drawing.Size(338, 218);
            this.gbImageProcessControl.ResumeLayout(false);
            this.gbImageProcessControl.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox gbImageProcessControl;
        private System.Windows.Forms.Button btnSendImgProcCtrlMsg;
        private System.Windows.Forms.CheckBox chkBxGPUTgtDetectionEnabled;
        private System.Windows.Forms.Label lbTgtDetectionType;
        private System.Windows.Forms.ComboBox cbTgtDetectionType;
        private System.Windows.Forms.Button btnImgProcEnabled;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbGPUStatus;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox tbActualTgtDetectType;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox tbVisionProcModeSelected;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.ComboBox cbVisionProcessorType;
    }
}
