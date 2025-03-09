namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class IMUControlWidget
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
            this.gBoxIMUControl = new System.Windows.Forms.GroupBox();
            this.tbIMUSerialCmd = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.btnSendCmd = new System.Windows.Forms.Button();
            this.tbCmdResponseMsg = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.chkbxRemoteCtrlEnable = new System.Windows.Forms.CheckBox();
            this.gBoxIMUControl.SuspendLayout();
            this.SuspendLayout();
            // 
            // gBoxIMUControl
            // 
            this.gBoxIMUControl.Controls.Add(this.chkbxRemoteCtrlEnable);
            this.gBoxIMUControl.Controls.Add(this.label2);
            this.gBoxIMUControl.Controls.Add(this.tbCmdResponseMsg);
            this.gBoxIMUControl.Controls.Add(this.btnSendCmd);
            this.gBoxIMUControl.Controls.Add(this.label1);
            this.gBoxIMUControl.Controls.Add(this.tbIMUSerialCmd);
            this.gBoxIMUControl.Location = new System.Drawing.Point(4, 0);
            this.gBoxIMUControl.Name = "gBoxIMUControl";
            this.gBoxIMUControl.Size = new System.Drawing.Size(350, 137);
            this.gBoxIMUControl.TabIndex = 0;
            this.gBoxIMUControl.TabStop = false;
            this.gBoxIMUControl.Text = "IMU Control";
            // 
            // tbIMUSerialCmd
            // 
            this.tbIMUSerialCmd.Location = new System.Drawing.Point(6, 55);
            this.tbIMUSerialCmd.Name = "tbIMUSerialCmd";
            this.tbIMUSerialCmd.Size = new System.Drawing.Size(238, 20);
            this.tbIMUSerialCmd.TabIndex = 0;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(6, 36);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(106, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "IMU Serial Command";
            // 
            // btnSendCmd
            // 
            this.btnSendCmd.Location = new System.Drawing.Point(259, 55);
            this.btnSendCmd.Name = "btnSendCmd";
            this.btnSendCmd.Size = new System.Drawing.Size(75, 23);
            this.btnSendCmd.TabIndex = 2;
            this.btnSendCmd.Text = "Send Cmd";
            this.btnSendCmd.UseVisualStyleBackColor = true;
            this.btnSendCmd.Click += new System.EventHandler(this.btnSendCmd_Click);
            // 
            // tbCmdResponseMsg
            // 
            this.tbCmdResponseMsg.Location = new System.Drawing.Point(6, 100);
            this.tbCmdResponseMsg.Name = "tbCmdResponseMsg";
            this.tbCmdResponseMsg.Size = new System.Drawing.Size(328, 20);
            this.tbCmdResponseMsg.TabIndex = 3;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(6, 84);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(105, 13);
            this.label2.TabIndex = 4;
            this.label2.Text = "Command Response";
            // 
            // chkbxRemoteCtrlEnable
            // 
            this.chkbxRemoteCtrlEnable.AutoSize = true;
            this.chkbxRemoteCtrlEnable.Location = new System.Drawing.Point(217, 19);
            this.chkbxRemoteCtrlEnable.Name = "chkbxRemoteCtrlEnable";
            this.chkbxRemoteCtrlEnable.Size = new System.Drawing.Size(117, 17);
            this.chkbxRemoteCtrlEnable.TabIndex = 5;
            this.chkbxRemoteCtrlEnable.Text = "Remote Ctrl Enable";
            this.chkbxRemoteCtrlEnable.UseVisualStyleBackColor = true;
            this.chkbxRemoteCtrlEnable.CheckedChanged += new System.EventHandler(this.chkbxRemoteCtrlEnable_CheckedChanged);
            // 
            // IMUControlWidget
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.gBoxIMUControl);
            this.Name = "IMUControlWidget";
            this.Size = new System.Drawing.Size(360, 142);
            this.gBoxIMUControl.ResumeLayout(false);
            this.gBoxIMUControl.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox gBoxIMUControl;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbCmdResponseMsg;
        private System.Windows.Forms.Button btnSendCmd;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox tbIMUSerialCmd;
        private System.Windows.Forms.CheckBox chkbxRemoteCtrlEnable;
    }
}
