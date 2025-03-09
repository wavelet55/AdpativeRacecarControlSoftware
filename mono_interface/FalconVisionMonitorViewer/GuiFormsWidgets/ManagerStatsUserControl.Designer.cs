namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class ManagerStatsUserControl
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
            this.groupBoxMgrStats = new System.Windows.Forms.GroupBox();
            this.cbxManagerName = new System.Windows.Forms.ComboBox();
            this.label16 = new System.Windows.Forms.Label();
            this.label15 = new System.Windows.Forms.Label();
            this.label14 = new System.Windows.Forms.Label();
            this.label13 = new System.Windows.Forms.Label();
            this.label12 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.tbTotalSleepTimeSec = new System.Windows.Forms.TextBox();
            this.tbAveSleepTimeSec = new System.Windows.Forms.TextBox();
            this.tbMaxSleepTimeSec = new System.Windows.Forms.TextBox();
            this.tbMinSleepTimeSec = new System.Windows.Forms.TextBox();
            this.tbTotalExecTimeSec = new System.Windows.Forms.TextBox();
            this.tbAveExecTimeSec = new System.Windows.Forms.TextBox();
            this.tbMaxExecTimeSec = new System.Windows.Forms.TextBox();
            this.tbMinExecTimeSec = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.tbNoWakeUpsAsleep = new System.Windows.Forms.TextBox();
            this.tbNoWakeUpsAwake = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.tbMgrExecTimeSec = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.tbNumExecCycles = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.btnResetStats = new System.Windows.Forms.Button();
            this.label5 = new System.Windows.Forms.Label();
            this.tbStatsUpdateTimeSec = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.tbMgrErrorCode = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.tbMgrRunningState = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.groupBoxMgrStats.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBoxMgrStats
            // 
            this.groupBoxMgrStats.Controls.Add(this.cbxManagerName);
            this.groupBoxMgrStats.Controls.Add(this.label16);
            this.groupBoxMgrStats.Controls.Add(this.label15);
            this.groupBoxMgrStats.Controls.Add(this.label14);
            this.groupBoxMgrStats.Controls.Add(this.label13);
            this.groupBoxMgrStats.Controls.Add(this.label12);
            this.groupBoxMgrStats.Controls.Add(this.label11);
            this.groupBoxMgrStats.Controls.Add(this.tbTotalSleepTimeSec);
            this.groupBoxMgrStats.Controls.Add(this.tbAveSleepTimeSec);
            this.groupBoxMgrStats.Controls.Add(this.tbMaxSleepTimeSec);
            this.groupBoxMgrStats.Controls.Add(this.tbMinSleepTimeSec);
            this.groupBoxMgrStats.Controls.Add(this.tbTotalExecTimeSec);
            this.groupBoxMgrStats.Controls.Add(this.tbAveExecTimeSec);
            this.groupBoxMgrStats.Controls.Add(this.tbMaxExecTimeSec);
            this.groupBoxMgrStats.Controls.Add(this.tbMinExecTimeSec);
            this.groupBoxMgrStats.Controls.Add(this.label10);
            this.groupBoxMgrStats.Controls.Add(this.label9);
            this.groupBoxMgrStats.Controls.Add(this.tbNoWakeUpsAsleep);
            this.groupBoxMgrStats.Controls.Add(this.tbNoWakeUpsAwake);
            this.groupBoxMgrStats.Controls.Add(this.label8);
            this.groupBoxMgrStats.Controls.Add(this.tbMgrExecTimeSec);
            this.groupBoxMgrStats.Controls.Add(this.label7);
            this.groupBoxMgrStats.Controls.Add(this.tbNumExecCycles);
            this.groupBoxMgrStats.Controls.Add(this.label6);
            this.groupBoxMgrStats.Controls.Add(this.btnResetStats);
            this.groupBoxMgrStats.Controls.Add(this.label5);
            this.groupBoxMgrStats.Controls.Add(this.tbStatsUpdateTimeSec);
            this.groupBoxMgrStats.Controls.Add(this.label4);
            this.groupBoxMgrStats.Controls.Add(this.tbMgrErrorCode);
            this.groupBoxMgrStats.Controls.Add(this.label3);
            this.groupBoxMgrStats.Controls.Add(this.tbMgrRunningState);
            this.groupBoxMgrStats.Controls.Add(this.label2);
            this.groupBoxMgrStats.Controls.Add(this.label1);
            this.groupBoxMgrStats.Location = new System.Drawing.Point(4, 4);
            this.groupBoxMgrStats.Name = "groupBoxMgrStats";
            this.groupBoxMgrStats.Size = new System.Drawing.Size(408, 267);
            this.groupBoxMgrStats.TabIndex = 0;
            this.groupBoxMgrStats.TabStop = false;
            this.groupBoxMgrStats.Text = "Manager Stats";
            // 
            // cbxManagerName
            // 
            this.cbxManagerName.FormattingEnabled = true;
            this.cbxManagerName.Location = new System.Drawing.Point(6, 42);
            this.cbxManagerName.Name = "cbxManagerName";
            this.cbxManagerName.Size = new System.Drawing.Size(158, 21);
            this.cbxManagerName.TabIndex = 33;
            this.cbxManagerName.SelectedIndexChanged += new System.EventHandler(this.cbxManagerName_SelectedIndexChanged);
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Location = new System.Drawing.Point(340, 185);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(31, 13);
            this.label16.TabIndex = 32;
            this.label16.Text = "Total";
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Location = new System.Drawing.Point(270, 185);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(26, 13);
            this.label15.TabIndex = 31;
            this.label15.Text = "Ave";
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(198, 185);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(27, 13);
            this.label14.TabIndex = 30;
            this.label14.Text = "Max";
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(123, 185);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(24, 13);
            this.label13.TabIndex = 29;
            this.label13.Text = "Min";
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(6, 230);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(91, 13);
            this.label12.TabIndex = 28;
            this.label12.Text = "Sleep Time (Sec):";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(9, 204);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(88, 13);
            this.label11.TabIndex = 27;
            this.label11.Text = "Exec Time (Sec):";
            // 
            // tbTotalSleepTimeSec
            // 
            this.tbTotalSleepTimeSec.Location = new System.Drawing.Point(326, 227);
            this.tbTotalSleepTimeSec.Name = "tbTotalSleepTimeSec";
            this.tbTotalSleepTimeSec.Size = new System.Drawing.Size(66, 20);
            this.tbTotalSleepTimeSec.TabIndex = 26;
            this.tbTotalSleepTimeSec.Text = "0";
            this.tbTotalSleepTimeSec.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // tbAveSleepTimeSec
            // 
            this.tbAveSleepTimeSec.Location = new System.Drawing.Point(254, 227);
            this.tbAveSleepTimeSec.Name = "tbAveSleepTimeSec";
            this.tbAveSleepTimeSec.Size = new System.Drawing.Size(66, 20);
            this.tbAveSleepTimeSec.TabIndex = 25;
            this.tbAveSleepTimeSec.Text = "0";
            this.tbAveSleepTimeSec.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // tbMaxSleepTimeSec
            // 
            this.tbMaxSleepTimeSec.Location = new System.Drawing.Point(182, 227);
            this.tbMaxSleepTimeSec.Name = "tbMaxSleepTimeSec";
            this.tbMaxSleepTimeSec.Size = new System.Drawing.Size(66, 20);
            this.tbMaxSleepTimeSec.TabIndex = 24;
            this.tbMaxSleepTimeSec.Text = "0";
            this.tbMaxSleepTimeSec.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // tbMinSleepTimeSec
            // 
            this.tbMinSleepTimeSec.Location = new System.Drawing.Point(110, 227);
            this.tbMinSleepTimeSec.Name = "tbMinSleepTimeSec";
            this.tbMinSleepTimeSec.Size = new System.Drawing.Size(66, 20);
            this.tbMinSleepTimeSec.TabIndex = 23;
            this.tbMinSleepTimeSec.Text = "0";
            this.tbMinSleepTimeSec.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // tbTotalExecTimeSec
            // 
            this.tbTotalExecTimeSec.Location = new System.Drawing.Point(326, 201);
            this.tbTotalExecTimeSec.Name = "tbTotalExecTimeSec";
            this.tbTotalExecTimeSec.Size = new System.Drawing.Size(66, 20);
            this.tbTotalExecTimeSec.TabIndex = 22;
            this.tbTotalExecTimeSec.Text = "0";
            this.tbTotalExecTimeSec.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // tbAveExecTimeSec
            // 
            this.tbAveExecTimeSec.Location = new System.Drawing.Point(254, 201);
            this.tbAveExecTimeSec.Name = "tbAveExecTimeSec";
            this.tbAveExecTimeSec.Size = new System.Drawing.Size(66, 20);
            this.tbAveExecTimeSec.TabIndex = 21;
            this.tbAveExecTimeSec.Text = "0";
            this.tbAveExecTimeSec.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // tbMaxExecTimeSec
            // 
            this.tbMaxExecTimeSec.Location = new System.Drawing.Point(182, 201);
            this.tbMaxExecTimeSec.Name = "tbMaxExecTimeSec";
            this.tbMaxExecTimeSec.Size = new System.Drawing.Size(66, 20);
            this.tbMaxExecTimeSec.TabIndex = 20;
            this.tbMaxExecTimeSec.Text = "0";
            this.tbMaxExecTimeSec.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // tbMinExecTimeSec
            // 
            this.tbMinExecTimeSec.Location = new System.Drawing.Point(110, 201);
            this.tbMinExecTimeSec.Name = "tbMinExecTimeSec";
            this.tbMinExecTimeSec.Size = new System.Drawing.Size(66, 20);
            this.tbMinExecTimeSec.TabIndex = 19;
            this.tbMinExecTimeSec.Text = "0";
            this.tbMinExecTimeSec.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(341, 135);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(39, 13);
            this.label10.TabIndex = 18;
            this.label10.Text = "Asleep";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(270, 136);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(40, 13);
            this.label9.TabIndex = 17;
            this.label9.Text = "Awake";
            // 
            // tbNoWakeUpsAsleep
            // 
            this.tbNoWakeUpsAsleep.Location = new System.Drawing.Point(326, 152);
            this.tbNoWakeUpsAsleep.Name = "tbNoWakeUpsAsleep";
            this.tbNoWakeUpsAsleep.Size = new System.Drawing.Size(66, 20);
            this.tbNoWakeUpsAsleep.TabIndex = 16;
            this.tbNoWakeUpsAsleep.Text = "0";
            this.tbNoWakeUpsAsleep.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // tbNoWakeUpsAwake
            // 
            this.tbNoWakeUpsAwake.Location = new System.Drawing.Point(254, 152);
            this.tbNoWakeUpsAwake.Name = "tbNoWakeUpsAwake";
            this.tbNoWakeUpsAwake.Size = new System.Drawing.Size(66, 20);
            this.tbNoWakeUpsAwake.TabIndex = 15;
            this.tbNoWakeUpsAwake.Text = "0";
            this.tbNoWakeUpsAwake.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(144, 155);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(104, 13);
            this.label8.TabIndex = 14;
            this.label8.Text = "Num Wake-up Calls:";
            // 
            // tbMgrExecTimeSec
            // 
            this.tbMgrExecTimeSec.Location = new System.Drawing.Point(326, 102);
            this.tbMgrExecTimeSec.Name = "tbMgrExecTimeSec";
            this.tbMgrExecTimeSec.Size = new System.Drawing.Size(66, 20);
            this.tbMgrExecTimeSec.TabIndex = 13;
            this.tbMgrExecTimeSec.Text = "0";
            this.tbMgrExecTimeSec.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(217, 105);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(103, 13);
            this.label7.TabIndex = 12;
            this.label7.Text = "Execute Time (Sec):";
            // 
            // tbNumExecCycles
            // 
            this.tbNumExecCycles.Location = new System.Drawing.Point(326, 76);
            this.tbNumExecCycles.Name = "tbNumExecCycles";
            this.tbNumExecCycles.Size = new System.Drawing.Size(66, 20);
            this.tbNumExecCycles.TabIndex = 11;
            this.tbNumExecCycles.Text = "0";
            this.tbNumExecCycles.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(227, 79);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(93, 13);
            this.label6.TabIndex = 10;
            this.label6.Text = "Num Exec Cycles:";
            // 
            // btnResetStats
            // 
            this.btnResetStats.Location = new System.Drawing.Point(12, 152);
            this.btnResetStats.Name = "btnResetStats";
            this.btnResetStats.Size = new System.Drawing.Size(75, 23);
            this.btnResetStats.TabIndex = 9;
            this.btnResetStats.Text = "Reset Stats";
            this.btnResetStats.UseVisualStyleBackColor = true;
            this.btnResetStats.Click += new System.EventHandler(this.btnResetStats_Click);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(323, 13);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(69, 13);
            this.label5.TabIndex = 8;
            this.label5.Text = "Stats Update";
            // 
            // tbStatsUpdateTimeSec
            // 
            this.tbStatsUpdateTimeSec.Location = new System.Drawing.Point(326, 43);
            this.tbStatsUpdateTimeSec.Name = "tbStatsUpdateTimeSec";
            this.tbStatsUpdateTimeSec.Size = new System.Drawing.Size(66, 20);
            this.tbStatsUpdateTimeSec.TabIndex = 7;
            this.tbStatsUpdateTimeSec.Text = "10.0";
            this.tbStatsUpdateTimeSec.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(323, 26);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(58, 13);
            this.label4.TabIndex = 6;
            this.label4.Text = "Time (Sec)";
            // 
            // tbMgrErrorCode
            // 
            this.tbMgrErrorCode.Location = new System.Drawing.Point(254, 43);
            this.tbMgrErrorCode.Name = "tbMgrErrorCode";
            this.tbMgrErrorCode.Size = new System.Drawing.Size(54, 20);
            this.tbMgrErrorCode.TabIndex = 5;
            this.tbMgrErrorCode.Text = "0";
            this.tbMgrErrorCode.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(251, 26);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(57, 13);
            this.label3.TabIndex = 4;
            this.label3.Text = "Error Code";
            // 
            // tbMgrRunningState
            // 
            this.tbMgrRunningState.Location = new System.Drawing.Point(176, 43);
            this.tbMgrRunningState.Name = "tbMgrRunningState";
            this.tbMgrRunningState.Size = new System.Drawing.Size(72, 20);
            this.tbMgrRunningState.TabIndex = 3;
            this.tbMgrRunningState.Text = "????";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(173, 26);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(75, 13);
            this.label2.TabIndex = 2;
            this.label2.Text = "Running State";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(6, 27);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(80, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Manager Name";
            // 
            // ManagerStatsUserControl
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.groupBoxMgrStats);
            this.Name = "ManagerStatsUserControl";
            this.Size = new System.Drawing.Size(420, 277);
            this.groupBoxMgrStats.ResumeLayout(false);
            this.groupBoxMgrStats.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBoxMgrStats;
        private System.Windows.Forms.TextBox tbMgrExecTimeSec;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox tbNumExecCycles;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Button btnResetStats;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox tbStatsUpdateTimeSec;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox tbMgrErrorCode;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbMgrRunningState;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label16;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.TextBox tbTotalSleepTimeSec;
        private System.Windows.Forms.TextBox tbAveSleepTimeSec;
        private System.Windows.Forms.TextBox tbMaxSleepTimeSec;
        private System.Windows.Forms.TextBox tbMinSleepTimeSec;
        private System.Windows.Forms.TextBox tbTotalExecTimeSec;
        private System.Windows.Forms.TextBox tbAveExecTimeSec;
        private System.Windows.Forms.TextBox tbMaxExecTimeSec;
        private System.Windows.Forms.TextBox tbMinExecTimeSec;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TextBox tbNoWakeUpsAsleep;
        private System.Windows.Forms.TextBox tbNoWakeUpsAwake;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.ComboBox cbxManagerName;
    }
}
