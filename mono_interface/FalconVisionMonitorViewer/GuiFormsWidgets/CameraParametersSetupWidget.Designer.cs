namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class CameraParametersSetupWidget
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
            this.gBoxCameraParametersSetup = new System.Windows.Forms.GroupBox();
            this.label1 = new System.Windows.Forms.Label();
            this.cbImageFormat = new System.Windows.Forms.ComboBox();
            this.lbImageHeight = new System.Windows.Forms.Label();
            this.tbImageHeight = new System.Windows.Forms.TextBox();
            this.lbImageWidth = new System.Windows.Forms.Label();
            this.tbImageWidth = new System.Windows.Forms.TextBox();
            this.btSendParameters = new System.Windows.Forms.Button();
            this.label6 = new System.Windows.Forms.Label();
            this.tbBrightnessValue = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.tbFocusValue = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.tbMode = new System.Windows.Forms.TextBox();
            this.gBoxCameraParametersSetup.SuspendLayout();
            this.SuspendLayout();
            // 
            // gBoxCameraParametersSetup
            // 
            this.gBoxCameraParametersSetup.Controls.Add(this.label2);
            this.gBoxCameraParametersSetup.Controls.Add(this.tbMode);
            this.gBoxCameraParametersSetup.Controls.Add(this.label6);
            this.gBoxCameraParametersSetup.Controls.Add(this.tbBrightnessValue);
            this.gBoxCameraParametersSetup.Controls.Add(this.label5);
            this.gBoxCameraParametersSetup.Controls.Add(this.tbFocusValue);
            this.gBoxCameraParametersSetup.Controls.Add(this.btSendParameters);
            this.gBoxCameraParametersSetup.Controls.Add(this.lbImageHeight);
            this.gBoxCameraParametersSetup.Controls.Add(this.tbImageHeight);
            this.gBoxCameraParametersSetup.Controls.Add(this.lbImageWidth);
            this.gBoxCameraParametersSetup.Controls.Add(this.tbImageWidth);
            this.gBoxCameraParametersSetup.Controls.Add(this.label1);
            this.gBoxCameraParametersSetup.Controls.Add(this.cbImageFormat);
            this.gBoxCameraParametersSetup.Location = new System.Drawing.Point(0, 4);
            this.gBoxCameraParametersSetup.Name = "gBoxCameraParametersSetup";
            this.gBoxCameraParametersSetup.Size = new System.Drawing.Size(251, 150);
            this.gBoxCameraParametersSetup.TabIndex = 0;
            this.gBoxCameraParametersSetup.TabStop = false;
            this.gBoxCameraParametersSetup.Text = "Camera Parameters Setup";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(18, 25);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(71, 13);
            this.label1.TabIndex = 3;
            this.label1.Text = "Image Format";
            // 
            // cbImageFormat
            // 
            this.cbImageFormat.FormattingEnabled = true;
            this.cbImageFormat.Location = new System.Drawing.Point(18, 41);
            this.cbImageFormat.Name = "cbImageFormat";
            this.cbImageFormat.Size = new System.Drawing.Size(121, 21);
            this.cbImageFormat.TabIndex = 2;
            // 
            // lbImageHeight
            // 
            this.lbImageHeight.AutoSize = true;
            this.lbImageHeight.Location = new System.Drawing.Point(111, 100);
            this.lbImageHeight.Name = "lbImageHeight";
            this.lbImageHeight.Size = new System.Drawing.Size(70, 13);
            this.lbImageHeight.TabIndex = 30;
            this.lbImageHeight.Text = "Image Height";
            // 
            // tbImageHeight
            // 
            this.tbImageHeight.Location = new System.Drawing.Point(111, 116);
            this.tbImageHeight.Name = "tbImageHeight";
            this.tbImageHeight.Size = new System.Drawing.Size(73, 20);
            this.tbImageHeight.TabIndex = 29;
            this.tbImageHeight.Text = "480";
            this.tbImageHeight.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // lbImageWidth
            // 
            this.lbImageWidth.AutoSize = true;
            this.lbImageWidth.Location = new System.Drawing.Point(18, 100);
            this.lbImageWidth.Name = "lbImageWidth";
            this.lbImageWidth.Size = new System.Drawing.Size(67, 13);
            this.lbImageWidth.TabIndex = 28;
            this.lbImageWidth.Text = "Image Width";
            // 
            // tbImageWidth
            // 
            this.tbImageWidth.Location = new System.Drawing.Point(18, 116);
            this.tbImageWidth.Name = "tbImageWidth";
            this.tbImageWidth.Size = new System.Drawing.Size(67, 20);
            this.tbImageWidth.TabIndex = 27;
            this.tbImageWidth.Text = "640";
            this.tbImageWidth.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // btSendParameters
            // 
            this.btSendParameters.Location = new System.Drawing.Point(155, 41);
            this.btSendParameters.Name = "btSendParameters";
            this.btSendParameters.Size = new System.Drawing.Size(75, 23);
            this.btSendParameters.TabIndex = 31;
            this.btSendParameters.Text = "Send";
            this.btSendParameters.TextImageRelation = System.Windows.Forms.TextImageRelation.TextBeforeImage;
            this.btSendParameters.UseVisualStyleBackColor = true;
            this.btSendParameters.Click += new System.EventHandler(this.btSendParameters_Click);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(59, 205);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(23, 13);
            this.label6.TabIndex = 36;
            this.label6.Text = "P-2";
            // 
            // tbBrightnessValue
            // 
            this.tbBrightnessValue.Location = new System.Drawing.Point(111, 202);
            this.tbBrightnessValue.Name = "tbBrightnessValue";
            this.tbBrightnessValue.Size = new System.Drawing.Size(73, 20);
            this.tbBrightnessValue.TabIndex = 35;
            this.tbBrightnessValue.Text = "0";
            this.tbBrightnessValue.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(59, 179);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(23, 13);
            this.label5.TabIndex = 34;
            this.label5.Text = "P-1";
            // 
            // tbFocusValue
            // 
            this.tbFocusValue.Location = new System.Drawing.Point(111, 176);
            this.tbFocusValue.Name = "tbFocusValue";
            this.tbFocusValue.Size = new System.Drawing.Size(73, 20);
            this.tbFocusValue.TabIndex = 32;
            this.tbFocusValue.Text = "0";
            this.tbFocusValue.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(26, 71);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(34, 13);
            this.label2.TabIndex = 38;
            this.label2.Text = "Mode";
            // 
            // tbMode
            // 
            this.tbMode.Location = new System.Drawing.Point(66, 68);
            this.tbMode.Name = "tbMode";
            this.tbMode.Size = new System.Drawing.Size(73, 20);
            this.tbMode.TabIndex = 37;
            this.tbMode.Text = "0";
            this.tbMode.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // CameraParametersSetupWidget
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.gBoxCameraParametersSetup);
            this.Name = "CameraParametersSetupWidget";
            this.Size = new System.Drawing.Size(257, 157);
            this.gBoxCameraParametersSetup.ResumeLayout(false);
            this.gBoxCameraParametersSetup.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox gBoxCameraParametersSetup;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.ComboBox cbImageFormat;
        private System.Windows.Forms.Label lbImageHeight;
        private System.Windows.Forms.TextBox tbImageHeight;
        private System.Windows.Forms.Label lbImageWidth;
        private System.Windows.Forms.TextBox tbImageWidth;
        private System.Windows.Forms.Button btSendParameters;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbMode;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox tbBrightnessValue;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox tbFocusValue;
    }
}
