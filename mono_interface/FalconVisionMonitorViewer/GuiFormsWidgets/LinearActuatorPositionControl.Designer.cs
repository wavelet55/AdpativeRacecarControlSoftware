namespace CarCANBusMonitor.Widgets
{
    partial class LinearActuatorPositionControl
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
            this.cbMotorEnFb = new System.Windows.Forms.CheckBox();
            this.cbClutchEnFb = new System.Windows.Forms.CheckBox();
            this.label2 = new System.Windows.Forms.Label();
            this.tbCtrlStatus = new System.Windows.Forms.TextBox();
            this.tbFunctionName = new System.Windows.Forms.TextBox();
            this.btnClearMaxMotorCurrent = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.tbMaxMotorCurrent = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.tbMotorCurrent = new System.Windows.Forms.TextBox();
            this.chkBxEnableMotor = new System.Windows.Forms.CheckBox();
            this.chkBxEnableClutch = new System.Windows.Forms.CheckBox();
            this.btnSendPos = new System.Windows.Forms.Button();
            this.hScrollBarPos = new System.Windows.Forms.HScrollBar();
            this.label3 = new System.Windows.Forms.Label();
            this.tbActualPosPercent = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.tbSetPositionPercent = new System.Windows.Forms.TextBox();
            this.cboxModeSelect = new System.Windows.Forms.ComboBox();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.cboxModeSelect);
            this.groupBox1.Controls.Add(this.cbMotorEnFb);
            this.groupBox1.Controls.Add(this.cbClutchEnFb);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.tbCtrlStatus);
            this.groupBox1.Controls.Add(this.tbFunctionName);
            this.groupBox1.Controls.Add(this.btnClearMaxMotorCurrent);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Controls.Add(this.tbMaxMotorCurrent);
            this.groupBox1.Controls.Add(this.label5);
            this.groupBox1.Controls.Add(this.tbMotorCurrent);
            this.groupBox1.Controls.Add(this.chkBxEnableMotor);
            this.groupBox1.Controls.Add(this.chkBxEnableClutch);
            this.groupBox1.Controls.Add(this.btnSendPos);
            this.groupBox1.Controls.Add(this.hScrollBarPos);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.tbActualPosPercent);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.tbSetPositionPercent);
            this.groupBox1.Location = new System.Drawing.Point(0, 4);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(343, 223);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Linear Actuater Position Control";
            // 
            // cbMotorEnFb
            // 
            this.cbMotorEnFb.AutoSize = true;
            this.cbMotorEnFb.Enabled = false;
            this.cbMotorEnFb.Location = new System.Drawing.Point(117, 199);
            this.cbMotorEnFb.Name = "cbMotorEnFb";
            this.cbMotorEnFb.Size = new System.Drawing.Size(95, 17);
            this.cbMotorEnFb.TabIndex = 25;
            this.cbMotorEnFb.Text = "Motor Enabled";
            this.cbMotorEnFb.UseVisualStyleBackColor = true;
            // 
            // cbClutchEnFb
            // 
            this.cbClutchEnFb.AutoSize = true;
            this.cbClutchEnFb.Enabled = false;
            this.cbClutchEnFb.Location = new System.Drawing.Point(117, 177);
            this.cbClutchEnFb.Name = "cbClutchEnFb";
            this.cbClutchEnFb.Size = new System.Drawing.Size(98, 17);
            this.cbClutchEnFb.TabIndex = 24;
            this.cbClutchEnFb.Text = "Clutch Enabled";
            this.cbClutchEnFb.UseVisualStyleBackColor = true;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(6, 178);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(37, 13);
            this.label2.TabIndex = 23;
            this.label2.Text = "Status";
            // 
            // tbCtrlStatus
            // 
            this.tbCtrlStatus.Location = new System.Drawing.Point(6, 197);
            this.tbCtrlStatus.Name = "tbCtrlStatus";
            this.tbCtrlStatus.ReadOnly = true;
            this.tbCtrlStatus.Size = new System.Drawing.Size(100, 20);
            this.tbCtrlStatus.TabIndex = 22;
            // 
            // tbFunctionName
            // 
            this.tbFunctionName.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.tbFunctionName.Font = new System.Drawing.Font("Microsoft Sans Serif", 16F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tbFunctionName.ForeColor = System.Drawing.SystemColors.MenuHighlight;
            this.tbFunctionName.Location = new System.Drawing.Point(105, 17);
            this.tbFunctionName.Name = "tbFunctionName";
            this.tbFunctionName.ReadOnly = true;
            this.tbFunctionName.Size = new System.Drawing.Size(100, 25);
            this.tbFunctionName.TabIndex = 20;
            this.tbFunctionName.Text = "Type";
            this.tbFunctionName.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // btnClearMaxMotorCurrent
            // 
            this.btnClearMaxMotorCurrent.Location = new System.Drawing.Point(247, 178);
            this.btnClearMaxMotorCurrent.Name = "btnClearMaxMotorCurrent";
            this.btnClearMaxMotorCurrent.Size = new System.Drawing.Size(75, 23);
            this.btnClearMaxMotorCurrent.TabIndex = 19;
            this.btnClearMaxMotorCurrent.Text = "Clear";
            this.btnClearMaxMotorCurrent.UseVisualStyleBackColor = true;
            this.btnClearMaxMotorCurrent.Click += new System.EventHandler(this.btnClearMaxMotorCurrent_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(230, 127);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(94, 13);
            this.label1.TabIndex = 18;
            this.label1.Text = "Max Motor Current";
            // 
            // tbMaxMotorCurrent
            // 
            this.tbMaxMotorCurrent.Location = new System.Drawing.Point(230, 146);
            this.tbMaxMotorCurrent.Name = "tbMaxMotorCurrent";
            this.tbMaxMotorCurrent.ReadOnly = true;
            this.tbMaxMotorCurrent.Size = new System.Drawing.Size(100, 20);
            this.tbMaxMotorCurrent.TabIndex = 17;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(117, 127);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(106, 13);
            this.label5.TabIndex = 14;
            this.label5.Text = "Motor Current (Amps)";
            // 
            // tbMotorCurrent
            // 
            this.tbMotorCurrent.Location = new System.Drawing.Point(117, 146);
            this.tbMotorCurrent.Name = "tbMotorCurrent";
            this.tbMotorCurrent.ReadOnly = true;
            this.tbMotorCurrent.Size = new System.Drawing.Size(100, 20);
            this.tbMotorCurrent.TabIndex = 13;
            // 
            // chkBxEnableMotor
            // 
            this.chkBxEnableMotor.AutoSize = true;
            this.chkBxEnableMotor.Location = new System.Drawing.Point(235, 63);
            this.chkBxEnableMotor.Name = "chkBxEnableMotor";
            this.chkBxEnableMotor.Size = new System.Drawing.Size(89, 17);
            this.chkBxEnableMotor.TabIndex = 11;
            this.chkBxEnableMotor.Text = "Enable Motor";
            this.chkBxEnableMotor.UseVisualStyleBackColor = true;
            this.chkBxEnableMotor.CheckedChanged += new System.EventHandler(this.chkBxEnableMotor_CheckedChanged);
            // 
            // chkBxEnableClutch
            // 
            this.chkBxEnableClutch.AutoSize = true;
            this.chkBxEnableClutch.Location = new System.Drawing.Point(235, 41);
            this.chkBxEnableClutch.Name = "chkBxEnableClutch";
            this.chkBxEnableClutch.Size = new System.Drawing.Size(92, 17);
            this.chkBxEnableClutch.TabIndex = 10;
            this.chkBxEnableClutch.Text = "Enable Clutch";
            this.chkBxEnableClutch.UseVisualStyleBackColor = true;
            this.chkBxEnableClutch.CheckedChanged += new System.EventHandler(this.chkBxEnableClutch_CheckedChanged);
            // 
            // btnSendPos
            // 
            this.btnSendPos.Location = new System.Drawing.Point(128, 57);
            this.btnSendPos.Name = "btnSendPos";
            this.btnSendPos.Size = new System.Drawing.Size(75, 23);
            this.btnSendPos.TabIndex = 9;
            this.btnSendPos.Text = "Send";
            this.btnSendPos.UseVisualStyleBackColor = true;
            this.btnSendPos.Click += new System.EventHandler(this.btnSendPos_Click);
            // 
            // hScrollBarPos
            // 
            this.hScrollBarPos.Cursor = System.Windows.Forms.Cursors.Cross;
            this.hScrollBarPos.Location = new System.Drawing.Point(9, 93);
            this.hScrollBarPos.Name = "hScrollBarPos";
            this.hScrollBarPos.Size = new System.Drawing.Size(324, 19);
            this.hScrollBarPos.TabIndex = 8;
            this.hScrollBarPos.Scroll += new System.Windows.Forms.ScrollEventHandler(this.hScrollBarPos_Scroll);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(6, 127);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(83, 13);
            this.label3.TabIndex = 7;
            this.label3.Text = "Act. Position (%)";
            // 
            // tbActualPosPercent
            // 
            this.tbActualPosPercent.Location = new System.Drawing.Point(6, 146);
            this.tbActualPosPercent.Name = "tbActualPosPercent";
            this.tbActualPosPercent.ReadOnly = true;
            this.tbActualPosPercent.Size = new System.Drawing.Size(100, 20);
            this.tbActualPosPercent.TabIndex = 6;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(6, 41);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(80, 13);
            this.label4.TabIndex = 5;
            this.label4.Text = "Set Position (%)";
            // 
            // tbSetPositionPercent
            // 
            this.tbSetPositionPercent.Location = new System.Drawing.Point(9, 59);
            this.tbSetPositionPercent.Name = "tbSetPositionPercent";
            this.tbSetPositionPercent.Size = new System.Drawing.Size(100, 20);
            this.tbSetPositionPercent.TabIndex = 4;
            // 
            // cboxModeSelect
            // 
            this.cboxModeSelect.FormattingEnabled = true;
            this.cboxModeSelect.Location = new System.Drawing.Point(227, 14);
            this.cboxModeSelect.Name = "cboxModeSelect";
            this.cboxModeSelect.Size = new System.Drawing.Size(95, 21);
            this.cboxModeSelect.TabIndex = 26;
            this.cboxModeSelect.SelectedIndexChanged += new System.EventHandler(this.cboxModeSelect_SelectedIndexChanged);
            // 
            // LinearActuatorPositionControl
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.groupBox1);
            this.Name = "LinearActuatorPositionControl";
            this.Size = new System.Drawing.Size(349, 230);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Button btnSendPos;
        private System.Windows.Forms.HScrollBar hScrollBarPos;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbActualPosPercent;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox tbSetPositionPercent;
        private System.Windows.Forms.CheckBox chkBxEnableMotor;
        private System.Windows.Forms.CheckBox chkBxEnableClutch;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox tbMotorCurrent;
        private System.Windows.Forms.Button btnClearMaxMotorCurrent;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox tbMaxMotorCurrent;
        private System.Windows.Forms.TextBox tbFunctionName;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbCtrlStatus;
        private System.Windows.Forms.CheckBox cbMotorEnFb;
        private System.Windows.Forms.CheckBox cbClutchEnFb;
        private System.Windows.Forms.ComboBox cboxModeSelect;
    }
}
