namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class VidereSystemStateControlWidget
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
            this.label6 = new System.Windows.Forms.Label();
            this.tbSystemState = new System.Windows.Forms.TextBox();
            this.cbSetSystemState = new System.Windows.Forms.ComboBox();
            this.chkBxDriverEnableSwitch = new System.Windows.Forms.CheckBox();
            this.chkBxHeadEnable = new System.Windows.Forms.CheckBox();
            this.chkBxThrottleEnable = new System.Windows.Forms.CheckBox();
            this.chkBxBrakeEnable = new System.Windows.Forms.CheckBox();
            this.chkBxBrakeControlFB = new System.Windows.Forms.CheckBox();
            this.chkBxThrottleControlFB = new System.Windows.Forms.CheckBox();
            this.chkBxHeadControlFB = new System.Windows.Forms.CheckBox();
            this.ckBox_BCI_En_State = new System.Windows.Forms.CheckBox();
            this.ckboxBCI_En_Ctrl = new System.Windows.Forms.CheckBox();
            this.textBoxBCI_ThottleCtrlState = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(0, 132);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(69, 13);
            this.label6.TabIndex = 21;
            this.label6.Text = "System State";
            // 
            // tbSystemState
            // 
            this.tbSystemState.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tbSystemState.Location = new System.Drawing.Point(3, 148);
            this.tbSystemState.Name = "tbSystemState";
            this.tbSystemState.ReadOnly = true;
            this.tbSystemState.Size = new System.Drawing.Size(128, 20);
            this.tbSystemState.TabIndex = 20;
            this.tbSystemState.Text = "Unknown";
            this.tbSystemState.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // cbSetSystemState
            // 
            this.cbSetSystemState.FormattingEnabled = true;
            this.cbSetSystemState.Location = new System.Drawing.Point(3, 175);
            this.cbSetSystemState.Name = "cbSetSystemState";
            this.cbSetSystemState.Size = new System.Drawing.Size(128, 21);
            this.cbSetSystemState.TabIndex = 19;
            this.cbSetSystemState.SelectedIndexChanged += new System.EventHandler(this.cbSetSystemState_SelectedIndexChanged);
            // 
            // chkBxDriverEnableSwitch
            // 
            this.chkBxDriverEnableSwitch.AutoSize = true;
            this.chkBxDriverEnableSwitch.Enabled = false;
            this.chkBxDriverEnableSwitch.Location = new System.Drawing.Point(4, 8);
            this.chkBxDriverEnableSwitch.Name = "chkBxDriverEnableSwitch";
            this.chkBxDriverEnableSwitch.Size = new System.Drawing.Size(88, 17);
            this.chkBxDriverEnableSwitch.TabIndex = 22;
            this.chkBxDriverEnableSwitch.Text = "Driver En Sw";
            this.chkBxDriverEnableSwitch.UseVisualStyleBackColor = true;
            // 
            // chkBxHeadEnable
            // 
            this.chkBxHeadEnable.AutoSize = true;
            this.chkBxHeadEnable.Location = new System.Drawing.Point(43, 31);
            this.chkBxHeadEnable.Name = "chkBxHeadEnable";
            this.chkBxHeadEnable.Size = new System.Drawing.Size(68, 17);
            this.chkBxHeadEnable.TabIndex = 23;
            this.chkBxHeadEnable.Text = "Head En";
            this.chkBxHeadEnable.UseVisualStyleBackColor = true;
            this.chkBxHeadEnable.CheckedChanged += new System.EventHandler(this.chkBxHeadEnable_CheckedChanged);
            // 
            // chkBxThrottleEnable
            // 
            this.chkBxThrottleEnable.AutoSize = true;
            this.chkBxThrottleEnable.Location = new System.Drawing.Point(43, 54);
            this.chkBxThrottleEnable.Name = "chkBxThrottleEnable";
            this.chkBxThrottleEnable.Size = new System.Drawing.Size(78, 17);
            this.chkBxThrottleEnable.TabIndex = 24;
            this.chkBxThrottleEnable.Text = "Throttle En";
            this.chkBxThrottleEnable.UseVisualStyleBackColor = true;
            this.chkBxThrottleEnable.CheckedChanged += new System.EventHandler(this.chkBxThrottleEnable_CheckedChanged);
            // 
            // chkBxBrakeEnable
            // 
            this.chkBxBrakeEnable.AutoSize = true;
            this.chkBxBrakeEnable.Location = new System.Drawing.Point(43, 77);
            this.chkBxBrakeEnable.Name = "chkBxBrakeEnable";
            this.chkBxBrakeEnable.Size = new System.Drawing.Size(70, 17);
            this.chkBxBrakeEnable.TabIndex = 25;
            this.chkBxBrakeEnable.Text = "Brake En";
            this.chkBxBrakeEnable.UseVisualStyleBackColor = true;
            this.chkBxBrakeEnable.CheckedChanged += new System.EventHandler(this.chkBxBrakeEnable_CheckedChanged);
            // 
            // chkBxBrakeControlFB
            // 
            this.chkBxBrakeControlFB.AutoSize = true;
            this.chkBxBrakeControlFB.Enabled = false;
            this.chkBxBrakeControlFB.Location = new System.Drawing.Point(4, 77);
            this.chkBxBrakeControlFB.Name = "chkBxBrakeControlFB";
            this.chkBxBrakeControlFB.Size = new System.Drawing.Size(15, 14);
            this.chkBxBrakeControlFB.TabIndex = 28;
            this.chkBxBrakeControlFB.UseVisualStyleBackColor = true;
            // 
            // chkBxThrottleControlFB
            // 
            this.chkBxThrottleControlFB.AutoSize = true;
            this.chkBxThrottleControlFB.Enabled = false;
            this.chkBxThrottleControlFB.Location = new System.Drawing.Point(4, 54);
            this.chkBxThrottleControlFB.Name = "chkBxThrottleControlFB";
            this.chkBxThrottleControlFB.Size = new System.Drawing.Size(15, 14);
            this.chkBxThrottleControlFB.TabIndex = 27;
            this.chkBxThrottleControlFB.UseVisualStyleBackColor = true;
            // 
            // chkBxHeadControlFB
            // 
            this.chkBxHeadControlFB.AutoSize = true;
            this.chkBxHeadControlFB.Enabled = false;
            this.chkBxHeadControlFB.Location = new System.Drawing.Point(4, 31);
            this.chkBxHeadControlFB.Name = "chkBxHeadControlFB";
            this.chkBxHeadControlFB.Size = new System.Drawing.Size(15, 14);
            this.chkBxHeadControlFB.TabIndex = 26;
            this.chkBxHeadControlFB.UseVisualStyleBackColor = true;
            // 
            // ckBox_BCI_En_State
            // 
            this.ckBox_BCI_En_State.AutoSize = true;
            this.ckBox_BCI_En_State.Enabled = false;
            this.ckBox_BCI_En_State.Location = new System.Drawing.Point(4, 100);
            this.ckBox_BCI_En_State.Name = "ckBox_BCI_En_State";
            this.ckBox_BCI_En_State.Size = new System.Drawing.Size(15, 14);
            this.ckBox_BCI_En_State.TabIndex = 30;
            this.ckBox_BCI_En_State.UseVisualStyleBackColor = true;
            // 
            // ckboxBCI_En_Ctrl
            // 
            this.ckboxBCI_En_Ctrl.AutoSize = true;
            this.ckboxBCI_En_Ctrl.Location = new System.Drawing.Point(43, 100);
            this.ckboxBCI_En_Ctrl.Name = "ckboxBCI_En_Ctrl";
            this.ckboxBCI_En_Ctrl.Size = new System.Drawing.Size(59, 17);
            this.ckboxBCI_En_Ctrl.TabIndex = 29;
            this.ckboxBCI_En_Ctrl.Text = "BCI En";
            this.ckboxBCI_En_Ctrl.UseVisualStyleBackColor = true;
            this.ckboxBCI_En_Ctrl.CheckedChanged += new System.EventHandler(this.ckboxBCI_En_Ctrl_CheckedChanged);
            // 
            // textBoxBCI_ThottleCtrlState
            // 
            this.textBoxBCI_ThottleCtrlState.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxBCI_ThottleCtrlState.Location = new System.Drawing.Point(4, 217);
            this.textBoxBCI_ThottleCtrlState.Name = "textBoxBCI_ThottleCtrlState";
            this.textBoxBCI_ThottleCtrlState.ReadOnly = true;
            this.textBoxBCI_ThottleCtrlState.Size = new System.Drawing.Size(128, 20);
            this.textBoxBCI_ThottleCtrlState.TabIndex = 31;
            this.textBoxBCI_ThottleCtrlState.Text = "Unknown";
            this.textBoxBCI_ThottleCtrlState.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(3, 201);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(52, 13);
            this.label1.TabIndex = 32;
            this.label1.Text = "BCI State";
            // 
            // VidereSystemStateControlWidget
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.label1);
            this.Controls.Add(this.textBoxBCI_ThottleCtrlState);
            this.Controls.Add(this.ckBox_BCI_En_State);
            this.Controls.Add(this.ckboxBCI_En_Ctrl);
            this.Controls.Add(this.chkBxBrakeControlFB);
            this.Controls.Add(this.chkBxThrottleControlFB);
            this.Controls.Add(this.chkBxHeadControlFB);
            this.Controls.Add(this.chkBxBrakeEnable);
            this.Controls.Add(this.chkBxThrottleEnable);
            this.Controls.Add(this.chkBxHeadEnable);
            this.Controls.Add(this.chkBxDriverEnableSwitch);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.tbSystemState);
            this.Controls.Add(this.cbSetSystemState);
            this.Name = "VidereSystemStateControlWidget";
            this.Size = new System.Drawing.Size(138, 246);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox tbSystemState;
        private System.Windows.Forms.ComboBox cbSetSystemState;
        private System.Windows.Forms.CheckBox chkBxDriverEnableSwitch;
        private System.Windows.Forms.CheckBox chkBxHeadEnable;
        private System.Windows.Forms.CheckBox chkBxThrottleEnable;
        private System.Windows.Forms.CheckBox chkBxBrakeEnable;
        private System.Windows.Forms.CheckBox chkBxBrakeControlFB;
        private System.Windows.Forms.CheckBox chkBxThrottleControlFB;
        private System.Windows.Forms.CheckBox chkBxHeadControlFB;
        private System.Windows.Forms.CheckBox ckBox_BCI_En_State;
        private System.Windows.Forms.CheckBox ckboxBCI_En_Ctrl;
        private System.Windows.Forms.TextBox textBoxBCI_ThottleCtrlState;
        private System.Windows.Forms.Label label1;
    }
}
