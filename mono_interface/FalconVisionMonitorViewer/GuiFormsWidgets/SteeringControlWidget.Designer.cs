namespace CarCANBusMonitor.Widgets
{
    partial class SteeringControlWidget
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
            this.components = new System.ComponentModel.Container();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.chkBxEnableSteeringCtrl = new System.Windows.Forms.CheckBox();
            this.chkBxFault = new System.Windows.Forms.CheckBox();
            this.chkBxHostCtrl = new System.Windows.Forms.CheckBox();
            this.chkBxMoveRight = new System.Windows.Forms.CheckBox();
            this.chkBxMoveLeft = new System.Windows.Forms.CheckBox();
            this.label7 = new System.Windows.Forms.Label();
            this.tbMotorVolts = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.tbMotorTorque = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.tbErrorCode = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.tbCtlrBoxTempC = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.tbMotorCurrent = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.tbMotorPWM = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.tbTorqueB = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.tbTorqueA = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.tbSteeringAngleDeg = new System.Windows.Forms.TextBox();
            this.hScrollBarPos = new System.Windows.Forms.HScrollBar();
            this.label1 = new System.Windows.Forms.Label();
            this.tbTorqueMapActual = new System.Windows.Forms.TextBox();
            this.cbTorqueMapSetting = new System.Windows.Forms.ComboBox();
            this.timerSendSteeringPos = new System.Windows.Forms.Timer(this.components);
            this.cbManualCtrlEnable = new System.Windows.Forms.CheckBox();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.cbManualCtrlEnable);
            this.groupBox1.Controls.Add(this.chkBxEnableSteeringCtrl);
            this.groupBox1.Controls.Add(this.chkBxFault);
            this.groupBox1.Controls.Add(this.chkBxHostCtrl);
            this.groupBox1.Controls.Add(this.chkBxMoveRight);
            this.groupBox1.Controls.Add(this.chkBxMoveLeft);
            this.groupBox1.Controls.Add(this.label7);
            this.groupBox1.Controls.Add(this.tbMotorVolts);
            this.groupBox1.Controls.Add(this.label8);
            this.groupBox1.Controls.Add(this.tbMotorTorque);
            this.groupBox1.Controls.Add(this.label9);
            this.groupBox1.Controls.Add(this.tbErrorCode);
            this.groupBox1.Controls.Add(this.label10);
            this.groupBox1.Controls.Add(this.tbCtlrBoxTempC);
            this.groupBox1.Controls.Add(this.label5);
            this.groupBox1.Controls.Add(this.tbMotorCurrent);
            this.groupBox1.Controls.Add(this.label6);
            this.groupBox1.Controls.Add(this.tbMotorPWM);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.tbTorqueB);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.tbTorqueA);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.tbSteeringAngleDeg);
            this.groupBox1.Controls.Add(this.hScrollBarPos);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Controls.Add(this.tbTorqueMapActual);
            this.groupBox1.Controls.Add(this.cbTorqueMapSetting);
            this.groupBox1.Location = new System.Drawing.Point(4, 4);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(361, 227);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Steering Control";
            // 
            // chkBxEnableSteeringCtrl
            // 
            this.chkBxEnableSteeringCtrl.AutoSize = true;
            this.chkBxEnableSteeringCtrl.Location = new System.Drawing.Point(14, 42);
            this.chkBxEnableSteeringCtrl.Name = "chkBxEnableSteeringCtrl";
            this.chkBxEnableSteeringCtrl.Size = new System.Drawing.Size(59, 17);
            this.chkBxEnableSteeringCtrl.TabIndex = 32;
            this.chkBxEnableSteeringCtrl.Text = "Enable";
            this.chkBxEnableSteeringCtrl.UseVisualStyleBackColor = true;
            this.chkBxEnableSteeringCtrl.CheckedChanged += new System.EventHandler(this.chkBxEnableSteeringCtrl_CheckedChanged);
            // 
            // chkBxFault
            // 
            this.chkBxFault.AutoSize = true;
            this.chkBxFault.Location = new System.Drawing.Point(258, 176);
            this.chkBxFault.Name = "chkBxFault";
            this.chkBxFault.Size = new System.Drawing.Size(49, 17);
            this.chkBxFault.TabIndex = 31;
            this.chkBxFault.Text = "Fault";
            this.chkBxFault.UseVisualStyleBackColor = true;
            // 
            // chkBxHostCtrl
            // 
            this.chkBxHostCtrl.AutoSize = true;
            this.chkBxHostCtrl.Location = new System.Drawing.Point(258, 153);
            this.chkBxHostCtrl.Name = "chkBxHostCtrl";
            this.chkBxHostCtrl.Size = new System.Drawing.Size(66, 17);
            this.chkBxHostCtrl.TabIndex = 30;
            this.chkBxHostCtrl.Text = "Host Ctrl";
            this.chkBxHostCtrl.UseVisualStyleBackColor = true;
            // 
            // chkBxMoveRight
            // 
            this.chkBxMoveRight.AutoSize = true;
            this.chkBxMoveRight.Location = new System.Drawing.Point(258, 130);
            this.chkBxMoveRight.Name = "chkBxMoveRight";
            this.chkBxMoveRight.Size = new System.Drawing.Size(75, 17);
            this.chkBxMoveRight.TabIndex = 29;
            this.chkBxMoveRight.Text = "Mov Right";
            this.chkBxMoveRight.UseVisualStyleBackColor = true;
            // 
            // chkBxMoveLeft
            // 
            this.chkBxMoveLeft.AutoSize = true;
            this.chkBxMoveLeft.Location = new System.Drawing.Point(258, 108);
            this.chkBxMoveLeft.Name = "chkBxMoveLeft";
            this.chkBxMoveLeft.Size = new System.Drawing.Size(68, 17);
            this.chkBxMoveLeft.TabIndex = 28;
            this.chkBxMoveLeft.Text = "Mov Left";
            this.chkBxMoveLeft.UseVisualStyleBackColor = true;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(129, 134);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(30, 13);
            this.label7.TabIndex = 27;
            this.label7.Text = "Volts";
            // 
            // tbMotorVolts
            // 
            this.tbMotorVolts.Location = new System.Drawing.Point(183, 131);
            this.tbMotorVolts.Name = "tbMotorVolts";
            this.tbMotorVolts.ReadOnly = true;
            this.tbMotorVolts.Size = new System.Drawing.Size(44, 20);
            this.tbMotorVolts.TabIndex = 26;
            this.tbMotorVolts.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(129, 108);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(41, 13);
            this.label8.TabIndex = 25;
            this.label8.Text = "Torque";
            // 
            // tbMotorTorque
            // 
            this.tbMotorTorque.Location = new System.Drawing.Point(183, 105);
            this.tbMotorTorque.Name = "tbMotorTorque";
            this.tbMotorTorque.ReadOnly = true;
            this.tbMotorTorque.Size = new System.Drawing.Size(44, 20);
            this.tbMotorTorque.TabIndex = 24;
            this.tbMotorTorque.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(129, 188);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(29, 13);
            this.label9.TabIndex = 23;
            this.label9.Text = "Error";
            // 
            // tbErrorCode
            // 
            this.tbErrorCode.Location = new System.Drawing.Point(183, 185);
            this.tbErrorCode.Name = "tbErrorCode";
            this.tbErrorCode.ReadOnly = true;
            this.tbErrorCode.Size = new System.Drawing.Size(44, 20);
            this.tbErrorCode.TabIndex = 22;
            this.tbErrorCode.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(129, 162);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(50, 13);
            this.label10.TabIndex = 21;
            this.label10.Text = "Temp (C)";
            // 
            // tbCtlrBoxTempC
            // 
            this.tbCtlrBoxTempC.Location = new System.Drawing.Point(183, 159);
            this.tbCtlrBoxTempC.Name = "tbCtlrBoxTempC";
            this.tbCtlrBoxTempC.ReadOnly = true;
            this.tbCtlrBoxTempC.Size = new System.Drawing.Size(44, 20);
            this.tbCtlrBoxTempC.TabIndex = 20;
            this.tbCtlrBoxTempC.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(8, 134);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(41, 13);
            this.label5.TabIndex = 19;
            this.label5.Text = "Current";
            // 
            // tbMotorCurrent
            // 
            this.tbMotorCurrent.Location = new System.Drawing.Point(62, 131);
            this.tbMotorCurrent.Name = "tbMotorCurrent";
            this.tbMotorCurrent.ReadOnly = true;
            this.tbMotorCurrent.Size = new System.Drawing.Size(44, 20);
            this.tbMotorCurrent.TabIndex = 18;
            this.tbMotorCurrent.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(8, 108);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(34, 13);
            this.label6.TabIndex = 17;
            this.label6.Text = "PWM";
            // 
            // tbMotorPWM
            // 
            this.tbMotorPWM.Location = new System.Drawing.Point(62, 105);
            this.tbMotorPWM.Name = "tbMotorPWM";
            this.tbMotorPWM.ReadOnly = true;
            this.tbMotorPWM.Size = new System.Drawing.Size(44, 20);
            this.tbMotorPWM.TabIndex = 16;
            this.tbMotorPWM.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(8, 188);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(48, 13);
            this.label4.TabIndex = 15;
            this.label4.Text = "TorqueB";
            // 
            // tbTorqueB
            // 
            this.tbTorqueB.Location = new System.Drawing.Point(62, 185);
            this.tbTorqueB.Name = "tbTorqueB";
            this.tbTorqueB.ReadOnly = true;
            this.tbTorqueB.Size = new System.Drawing.Size(44, 20);
            this.tbTorqueB.TabIndex = 14;
            this.tbTorqueB.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(8, 162);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(48, 13);
            this.label3.TabIndex = 13;
            this.label3.Text = "TorqueA";
            // 
            // tbTorqueA
            // 
            this.tbTorqueA.Location = new System.Drawing.Point(62, 159);
            this.tbTorqueA.Name = "tbTorqueA";
            this.tbTorqueA.ReadOnly = true;
            this.tbTorqueA.Size = new System.Drawing.Size(44, 20);
            this.tbTorqueA.TabIndex = 12;
            this.tbTorqueA.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(111, 20);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(105, 13);
            this.label2.TabIndex = 11;
            this.label2.Text = "Steering Angle (Deg)";
            // 
            // tbSteeringAngleDeg
            // 
            this.tbSteeringAngleDeg.Location = new System.Drawing.Point(114, 36);
            this.tbSteeringAngleDeg.Name = "tbSteeringAngleDeg";
            this.tbSteeringAngleDeg.ReadOnly = true;
            this.tbSteeringAngleDeg.Size = new System.Drawing.Size(102, 20);
            this.tbSteeringAngleDeg.TabIndex = 10;
            this.tbSteeringAngleDeg.Text = "0";
            this.tbSteeringAngleDeg.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // hScrollBarPos
            // 
            this.hScrollBarPos.Cursor = System.Windows.Forms.Cursors.Cross;
            this.hScrollBarPos.Location = new System.Drawing.Point(14, 70);
            this.hScrollBarPos.Minimum = -100;
            this.hScrollBarPos.Name = "hScrollBarPos";
            this.hScrollBarPos.Size = new System.Drawing.Size(324, 19);
            this.hScrollBarPos.TabIndex = 9;
            this.hScrollBarPos.Scroll += new System.Windows.Forms.ScrollEventHandler(this.hScrollBarPos_Scroll);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(222, 16);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(65, 13);
            this.label1.TabIndex = 2;
            this.label1.Text = "Torque Map";
            // 
            // tbTorqueMapActual
            // 
            this.tbTorqueMapActual.Location = new System.Drawing.Point(294, 36);
            this.tbTorqueMapActual.Name = "tbTorqueMapActual";
            this.tbTorqueMapActual.ReadOnly = true;
            this.tbTorqueMapActual.Size = new System.Drawing.Size(44, 20);
            this.tbTorqueMapActual.TabIndex = 1;
            // 
            // cbTorqueMapSetting
            // 
            this.cbTorqueMapSetting.FormattingEnabled = true;
            this.cbTorqueMapSetting.Location = new System.Drawing.Point(222, 35);
            this.cbTorqueMapSetting.Name = "cbTorqueMapSetting";
            this.cbTorqueMapSetting.Size = new System.Drawing.Size(65, 21);
            this.cbTorqueMapSetting.TabIndex = 0;
            this.cbTorqueMapSetting.SelectedIndexChanged += new System.EventHandler(this.cbTorqueMapSetting_SelectedIndexChanged);
            // 
            // timerSendSteeringPos
            // 
            this.timerSendSteeringPos.Tick += new System.EventHandler(this.timerSendSteeringPos_Tick);
            // 
            // cbManualCtrlEnable
            // 
            this.cbManualCtrlEnable.AutoSize = true;
            this.cbManualCtrlEnable.Location = new System.Drawing.Point(14, 19);
            this.cbManualCtrlEnable.Name = "cbManualCtrlEnable";
            this.cbManualCtrlEnable.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.cbManualCtrlEnable.Size = new System.Drawing.Size(79, 17);
            this.cbManualCtrlEnable.TabIndex = 33;
            this.cbManualCtrlEnable.Text = "Manual Ctrl";
            this.cbManualCtrlEnable.UseVisualStyleBackColor = true;
            this.cbManualCtrlEnable.CheckedChanged += new System.EventHandler(this.cbManualCtrlEnable_CheckedChanged);
            // 
            // SteeringControlWidget
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.groupBox1);
            this.Name = "SteeringControlWidget";
            this.Size = new System.Drawing.Size(373, 237);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox tbTorqueMapActual;
        private System.Windows.Forms.ComboBox cbTorqueMapSetting;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox tbMotorCurrent;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox tbMotorPWM;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox tbTorqueB;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbTorqueA;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbSteeringAngleDeg;
        private System.Windows.Forms.HScrollBar hScrollBarPos;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox tbMotorVolts;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox tbMotorTorque;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TextBox tbErrorCode;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.TextBox tbCtlrBoxTempC;
        private System.Windows.Forms.CheckBox chkBxFault;
        private System.Windows.Forms.CheckBox chkBxHostCtrl;
        private System.Windows.Forms.CheckBox chkBxMoveRight;
        private System.Windows.Forms.CheckBox chkBxMoveLeft;
        private System.Windows.Forms.CheckBox chkBxEnableSteeringCtrl;
        private System.Windows.Forms.Timer timerSendSteeringPos;
        private System.Windows.Forms.CheckBox cbManualCtrlEnable;
    }
}
