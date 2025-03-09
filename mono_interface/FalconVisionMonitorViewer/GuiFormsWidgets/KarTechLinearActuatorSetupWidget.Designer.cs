namespace CarCANBusMonitor.Widgets
{
    partial class KarTechLinearActuatorSetupWidget
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
            this.groupBoxLASetup = new System.Windows.Forms.GroupBox();
            this.tbActuatorFunction = new System.Windows.Forms.TextBox();
            this.btnReadConfigParameters = new System.Windows.Forms.Button();
            this.btnSetConfigParameters = new System.Windows.Forms.Button();
            this.btnResetConfigs = new System.Windows.Forms.Button();
            this.tbFeedbackLoopDeadband = new System.Windows.Forms.TextBox();
            this.label14 = new System.Windows.Forms.Label();
            this.tbFeedbackLoopCLFreq = new System.Windows.Forms.TextBox();
            this.label15 = new System.Windows.Forms.Label();
            this.tbMotorMaxPwmPercent = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.tbMotorMinPwmPercent = new System.Windows.Forms.TextBox();
            this.label11 = new System.Windows.Forms.Label();
            this.tbMotorPWMFreq = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.tbFeedbackLoop_KD = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.tbFeedbackLoop_KI = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.tbFeedbackLoop_KP = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.tbPosReachedErrTime = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.tbMotorMaxCurrentSetting = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.tbMaxPosition = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.tbMinPosition = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label13 = new System.Windows.Forms.Label();
            this.label12 = new System.Windows.Forms.Label();
            this.btnAutoZeroPosSensor = new System.Windows.Forms.Button();
            this.label17 = new System.Windows.Forms.Label();
            this.btnSetCmdRptIDs = new System.Windows.Forms.Button();
            this.label16 = new System.Windows.Forms.Label();
            this.btnResetOutput = new System.Windows.Forms.Button();
            this.label18 = new System.Windows.Forms.Label();
            this.btnResetAllCfgs = new System.Windows.Forms.Button();
            this.label19 = new System.Windows.Forms.Label();
            this.btnResetUserConfigs = new System.Windows.Forms.Button();
            this.label20 = new System.Windows.Forms.Label();
            this.btnResetHdwrCfgs = new System.Windows.Forms.Button();
            this.gpSetupControls = new System.Windows.Forms.GroupBox();
            this.label21 = new System.Windows.Forms.Label();
            this.groupBoxLASetup.SuspendLayout();
            this.gpSetupControls.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBoxLASetup
            // 
            this.groupBoxLASetup.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(255)))), ((int)(((byte)(192)))));
            this.groupBoxLASetup.Controls.Add(this.tbActuatorFunction);
            this.groupBoxLASetup.Controls.Add(this.btnReadConfigParameters);
            this.groupBoxLASetup.Controls.Add(this.btnSetConfigParameters);
            this.groupBoxLASetup.Controls.Add(this.btnResetConfigs);
            this.groupBoxLASetup.Controls.Add(this.tbFeedbackLoopDeadband);
            this.groupBoxLASetup.Controls.Add(this.label14);
            this.groupBoxLASetup.Controls.Add(this.tbFeedbackLoopCLFreq);
            this.groupBoxLASetup.Controls.Add(this.label15);
            this.groupBoxLASetup.Controls.Add(this.tbMotorMaxPwmPercent);
            this.groupBoxLASetup.Controls.Add(this.label10);
            this.groupBoxLASetup.Controls.Add(this.tbMotorMinPwmPercent);
            this.groupBoxLASetup.Controls.Add(this.label11);
            this.groupBoxLASetup.Controls.Add(this.tbMotorPWMFreq);
            this.groupBoxLASetup.Controls.Add(this.label8);
            this.groupBoxLASetup.Controls.Add(this.tbFeedbackLoop_KD);
            this.groupBoxLASetup.Controls.Add(this.label9);
            this.groupBoxLASetup.Controls.Add(this.tbFeedbackLoop_KI);
            this.groupBoxLASetup.Controls.Add(this.label6);
            this.groupBoxLASetup.Controls.Add(this.tbFeedbackLoop_KP);
            this.groupBoxLASetup.Controls.Add(this.label7);
            this.groupBoxLASetup.Controls.Add(this.tbPosReachedErrTime);
            this.groupBoxLASetup.Controls.Add(this.label5);
            this.groupBoxLASetup.Controls.Add(this.tbMotorMaxCurrentSetting);
            this.groupBoxLASetup.Controls.Add(this.label4);
            this.groupBoxLASetup.Controls.Add(this.label3);
            this.groupBoxLASetup.Controls.Add(this.tbMaxPosition);
            this.groupBoxLASetup.Controls.Add(this.label2);
            this.groupBoxLASetup.Controls.Add(this.tbMinPosition);
            this.groupBoxLASetup.Controls.Add(this.label1);
            this.groupBoxLASetup.Location = new System.Drawing.Point(4, 4);
            this.groupBoxLASetup.Name = "groupBoxLASetup";
            this.groupBoxLASetup.Size = new System.Drawing.Size(286, 481);
            this.groupBoxLASetup.TabIndex = 0;
            this.groupBoxLASetup.TabStop = false;
            this.groupBoxLASetup.Text = "KarTech Actuator Setup";
            this.groupBoxLASetup.Enter += new System.EventHandler(this.groupBoxLASetup_Enter);
            // 
            // tbActuatorFunction
            // 
            this.tbActuatorFunction.Location = new System.Drawing.Point(9, 41);
            this.tbActuatorFunction.Name = "tbActuatorFunction";
            this.tbActuatorFunction.ReadOnly = true;
            this.tbActuatorFunction.Size = new System.Drawing.Size(102, 20);
            this.tbActuatorFunction.TabIndex = 33;
            this.tbActuatorFunction.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // btnReadConfigParameters
            // 
            this.btnReadConfigParameters.Location = new System.Drawing.Point(10, 422);
            this.btnReadConfigParameters.Name = "btnReadConfigParameters";
            this.btnReadConfigParameters.Size = new System.Drawing.Size(92, 23);
            this.btnReadConfigParameters.TabIndex = 32;
            this.btnReadConfigParameters.Text = "Read Confgs";
            this.btnReadConfigParameters.UseVisualStyleBackColor = true;
            this.btnReadConfigParameters.Click += new System.EventHandler(this.btnReadConfigParameters_Click);
            // 
            // btnSetConfigParameters
            // 
            this.btnSetConfigParameters.Location = new System.Drawing.Point(179, 422);
            this.btnSetConfigParameters.Name = "btnSetConfigParameters";
            this.btnSetConfigParameters.Size = new System.Drawing.Size(92, 23);
            this.btnSetConfigParameters.TabIndex = 31;
            this.btnSetConfigParameters.Text = "Set Confgs";
            this.btnSetConfigParameters.UseVisualStyleBackColor = true;
            this.btnSetConfigParameters.Click += new System.EventHandler(this.btnSetConfigParameters_Click);
            // 
            // btnResetConfigs
            // 
            this.btnResetConfigs.Location = new System.Drawing.Point(179, 451);
            this.btnResetConfigs.Name = "btnResetConfigs";
            this.btnResetConfigs.Size = new System.Drawing.Size(92, 23);
            this.btnResetConfigs.TabIndex = 30;
            this.btnResetConfigs.Text = "Reset Confgs";
            this.btnResetConfigs.UseVisualStyleBackColor = true;
            this.btnResetConfigs.Click += new System.EventHandler(this.btnResetConfigs_Click);
            // 
            // tbFeedbackLoopDeadband
            // 
            this.tbFeedbackLoopDeadband.Location = new System.Drawing.Point(169, 295);
            this.tbFeedbackLoopDeadband.Name = "tbFeedbackLoopDeadband";
            this.tbFeedbackLoopDeadband.Size = new System.Drawing.Size(102, 20);
            this.tbFeedbackLoopDeadband.TabIndex = 29;
            this.tbFeedbackLoopDeadband.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(10, 298);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(149, 13);
            this.label14.TabIndex = 28;
            this.label14.Text = "Feedback Deadband (Inches)";
            // 
            // tbFeedbackLoopCLFreq
            // 
            this.tbFeedbackLoopCLFreq.Location = new System.Drawing.Point(169, 269);
            this.tbFeedbackLoopCLFreq.Name = "tbFeedbackLoopCLFreq";
            this.tbFeedbackLoopCLFreq.Size = new System.Drawing.Size(102, 20);
            this.tbFeedbackLoopCLFreq.TabIndex = 27;
            this.tbFeedbackLoopCLFreq.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Location = new System.Drawing.Point(10, 272);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(139, 13);
            this.label15.TabIndex = 26;
            this.label15.Text = "Feedback Update Freq (Hz)";
            // 
            // tbMotorMaxPwmPercent
            // 
            this.tbMotorMaxPwmPercent.Location = new System.Drawing.Point(169, 386);
            this.tbMotorMaxPwmPercent.Name = "tbMotorMaxPwmPercent";
            this.tbMotorMaxPwmPercent.Size = new System.Drawing.Size(102, 20);
            this.tbMotorMaxPwmPercent.TabIndex = 22;
            this.tbMotorMaxPwmPercent.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(10, 389);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(104, 13);
            this.label10.TabIndex = 21;
            this.label10.Text = "Motor PWM Max (%)";
            // 
            // tbMotorMinPwmPercent
            // 
            this.tbMotorMinPwmPercent.Location = new System.Drawing.Point(169, 360);
            this.tbMotorMinPwmPercent.Name = "tbMotorMinPwmPercent";
            this.tbMotorMinPwmPercent.Size = new System.Drawing.Size(102, 20);
            this.tbMotorMinPwmPercent.TabIndex = 20;
            this.tbMotorMinPwmPercent.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(10, 363);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(101, 13);
            this.label11.TabIndex = 19;
            this.label11.Text = "Motor PWM Min (%)";
            // 
            // tbMotorPWMFreq
            // 
            this.tbMotorPWMFreq.Location = new System.Drawing.Point(169, 334);
            this.tbMotorPWMFreq.Name = "tbMotorPWMFreq";
            this.tbMotorPWMFreq.Size = new System.Drawing.Size(102, 20);
            this.tbMotorPWMFreq.TabIndex = 18;
            this.tbMotorPWMFreq.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(10, 337);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(110, 13);
            this.label8.TabIndex = 17;
            this.label8.Text = "Motor PWM Freq (Hz)";
            // 
            // tbFeedbackLoop_KD
            // 
            this.tbFeedbackLoop_KD.Location = new System.Drawing.Point(169, 244);
            this.tbFeedbackLoop_KD.Name = "tbFeedbackLoop_KD";
            this.tbFeedbackLoop_KD.Size = new System.Drawing.Size(102, 20);
            this.tbFeedbackLoop_KD.TabIndex = 16;
            this.tbFeedbackLoop_KD.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(10, 247);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(100, 13);
            this.label9.TabIndex = 15;
            this.label9.Text = "Feedback Loop KD";
            // 
            // tbFeedbackLoop_KI
            // 
            this.tbFeedbackLoop_KI.Location = new System.Drawing.Point(169, 218);
            this.tbFeedbackLoop_KI.Name = "tbFeedbackLoop_KI";
            this.tbFeedbackLoop_KI.Size = new System.Drawing.Size(102, 20);
            this.tbFeedbackLoop_KI.TabIndex = 14;
            this.tbFeedbackLoop_KI.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(10, 221);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(95, 13);
            this.label6.TabIndex = 13;
            this.label6.Text = "Feedback Loop KI";
            // 
            // tbFeedbackLoop_KP
            // 
            this.tbFeedbackLoop_KP.Location = new System.Drawing.Point(169, 192);
            this.tbFeedbackLoop_KP.Name = "tbFeedbackLoop_KP";
            this.tbFeedbackLoop_KP.Size = new System.Drawing.Size(102, 20);
            this.tbFeedbackLoop_KP.TabIndex = 12;
            this.tbFeedbackLoop_KP.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(10, 195);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(99, 13);
            this.label7.TabIndex = 11;
            this.label7.Text = "Feedback Loop KP";
            // 
            // tbPosReachedErrTime
            // 
            this.tbPosReachedErrTime.Location = new System.Drawing.Point(169, 166);
            this.tbPosReachedErrTime.Name = "tbPosReachedErrTime";
            this.tbPosReachedErrTime.Size = new System.Drawing.Size(102, 20);
            this.tbPosReachedErrTime.TabIndex = 10;
            this.tbPosReachedErrTime.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(10, 169);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(157, 13);
            this.label5.TabIndex = 9;
            this.label5.Text = "Pos Reached Error Time (msec)";
            // 
            // tbMotorMaxCurrentSetting
            // 
            this.tbMotorMaxCurrentSetting.Location = new System.Drawing.Point(169, 140);
            this.tbMotorMaxCurrentSetting.Name = "tbMotorMaxCurrentSetting";
            this.tbMotorMaxCurrentSetting.ReadOnly = true;
            this.tbMotorMaxCurrentSetting.Size = new System.Drawing.Size(102, 20);
            this.tbMotorMaxCurrentSetting.TabIndex = 8;
            this.tbMotorMaxCurrentSetting.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(10, 143);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(129, 13);
            this.label4.TabIndex = 7;
            this.label4.Text = "Motor Max Current (Amps)";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(169, 80);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(89, 13);
            this.label3.TabIndex = 6;
            this.label3.Text = "Max Pos (Inches)";
            // 
            // tbMaxPosition
            // 
            this.tbMaxPosition.Location = new System.Drawing.Point(169, 96);
            this.tbMaxPosition.Name = "tbMaxPosition";
            this.tbMaxPosition.Size = new System.Drawing.Size(102, 20);
            this.tbMaxPosition.TabIndex = 5;
            this.tbMaxPosition.Text = "3.0";
            this.tbMaxPosition.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(10, 80);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(86, 13);
            this.label2.TabIndex = 4;
            this.label2.Text = "Min Pos (Inches)";
            // 
            // tbMinPosition
            // 
            this.tbMinPosition.Location = new System.Drawing.Point(10, 96);
            this.tbMinPosition.Name = "tbMinPosition";
            this.tbMinPosition.Size = new System.Drawing.Size(102, 20);
            this.tbMinPosition.TabIndex = 3;
            this.tbMinPosition.Text = "0";
            this.tbMinPosition.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(7, 25);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(120, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Actuator Function/Type";
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(10, 200);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(141, 13);
            this.label13.TabIndex = 36;
            this.label13.Text = "(Shaft Must be free to move)";
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(10, 183);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(130, 13);
            this.label12.TabIndex = 35;
            this.label12.Text = "Auto Zero Position Sensor";
            // 
            // btnAutoZeroPosSensor
            // 
            this.btnAutoZeroPosSensor.Location = new System.Drawing.Point(13, 216);
            this.btnAutoZeroPosSensor.Name = "btnAutoZeroPosSensor";
            this.btnAutoZeroPosSensor.Size = new System.Drawing.Size(92, 23);
            this.btnAutoZeroPosSensor.TabIndex = 34;
            this.btnAutoZeroPosSensor.Text = "Auto Zero";
            this.btnAutoZeroPosSensor.UseVisualStyleBackColor = true;
            this.btnAutoZeroPosSensor.Click += new System.EventHandler(this.btnAutoZeroPosSensor_Click);
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.Location = new System.Drawing.Point(186, 200);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(67, 13);
            this.label17.TabIndex = 38;
            this.label17.Text = "Set CAN IDs";
            // 
            // btnSetCmdRptIDs
            // 
            this.btnSetCmdRptIDs.Location = new System.Drawing.Point(174, 216);
            this.btnSetCmdRptIDs.Name = "btnSetCmdRptIDs";
            this.btnSetCmdRptIDs.Size = new System.Drawing.Size(92, 23);
            this.btnSetCmdRptIDs.TabIndex = 37;
            this.btnSetCmdRptIDs.Text = "Set CAN IDs";
            this.btnSetCmdRptIDs.UseVisualStyleBackColor = true;
            this.btnSetCmdRptIDs.Click += new System.EventHandler(this.btnSetCmdRptIDs_Click);
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Location = new System.Drawing.Point(186, 80);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(70, 13);
            this.label16.TabIndex = 40;
            this.label16.Text = "Reset Output";
            // 
            // btnResetOutput
            // 
            this.btnResetOutput.Location = new System.Drawing.Point(174, 96);
            this.btnResetOutput.Name = "btnResetOutput";
            this.btnResetOutput.Size = new System.Drawing.Size(92, 23);
            this.btnResetOutput.TabIndex = 39;
            this.btnResetOutput.Text = "Reset Output";
            this.btnResetOutput.UseVisualStyleBackColor = true;
            this.btnResetOutput.Click += new System.EventHandler(this.btnResetOutput_Click);
            // 
            // label18
            // 
            this.label18.AutoSize = true;
            this.label18.Location = new System.Drawing.Point(173, 131);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(93, 13);
            this.label18.TabIndex = 42;
            this.label18.Text = "Reset All Setttings";
            // 
            // btnResetAllCfgs
            // 
            this.btnResetAllCfgs.Location = new System.Drawing.Point(174, 147);
            this.btnResetAllCfgs.Name = "btnResetAllCfgs";
            this.btnResetAllCfgs.Size = new System.Drawing.Size(92, 23);
            this.btnResetAllCfgs.TabIndex = 41;
            this.btnResetAllCfgs.Text = "Reset All";
            this.btnResetAllCfgs.UseVisualStyleBackColor = true;
            this.btnResetAllCfgs.Click += new System.EventHandler(this.btnResetAllCfgs_Click);
            // 
            // label19
            // 
            this.label19.AutoSize = true;
            this.label19.Location = new System.Drawing.Point(12, 127);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(98, 13);
            this.label19.TabIndex = 44;
            this.label19.Text = "Reset User Configs";
            // 
            // btnResetUserConfigs
            // 
            this.btnResetUserConfigs.Location = new System.Drawing.Point(13, 143);
            this.btnResetUserConfigs.Name = "btnResetUserConfigs";
            this.btnResetUserConfigs.Size = new System.Drawing.Size(92, 23);
            this.btnResetUserConfigs.TabIndex = 43;
            this.btnResetUserConfigs.Text = "Reset User Cfgs";
            this.btnResetUserConfigs.UseVisualStyleBackColor = true;
            this.btnResetUserConfigs.Click += new System.EventHandler(this.btnResetUserConfigs_Click);
            // 
            // label20
            // 
            this.label20.AutoSize = true;
            this.label20.Location = new System.Drawing.Point(15, 80);
            this.label20.Name = "label20";
            this.label20.Size = new System.Drawing.Size(122, 13);
            this.label20.TabIndex = 46;
            this.label20.Text = "Reset Hardware Configs";
            // 
            // btnResetHdwrCfgs
            // 
            this.btnResetHdwrCfgs.Location = new System.Drawing.Point(16, 96);
            this.btnResetHdwrCfgs.Name = "btnResetHdwrCfgs";
            this.btnResetHdwrCfgs.Size = new System.Drawing.Size(92, 23);
            this.btnResetHdwrCfgs.TabIndex = 45;
            this.btnResetHdwrCfgs.Text = "Reset Hdwr Cfgs";
            this.btnResetHdwrCfgs.UseVisualStyleBackColor = true;
            this.btnResetHdwrCfgs.Click += new System.EventHandler(this.btnResetHdwrCfgs_Click);
            // 
            // gpSetupControls
            // 
            this.gpSetupControls.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(128)))), ((int)(((byte)(0)))));
            this.gpSetupControls.Controls.Add(this.label21);
            this.gpSetupControls.Controls.Add(this.label18);
            this.gpSetupControls.Controls.Add(this.label20);
            this.gpSetupControls.Controls.Add(this.btnResetAllCfgs);
            this.gpSetupControls.Controls.Add(this.label16);
            this.gpSetupControls.Controls.Add(this.btnAutoZeroPosSensor);
            this.gpSetupControls.Controls.Add(this.btnResetOutput);
            this.gpSetupControls.Controls.Add(this.btnResetHdwrCfgs);
            this.gpSetupControls.Controls.Add(this.label19);
            this.gpSetupControls.Controls.Add(this.label12);
            this.gpSetupControls.Controls.Add(this.btnResetUserConfigs);
            this.gpSetupControls.Controls.Add(this.label13);
            this.gpSetupControls.Controls.Add(this.btnSetCmdRptIDs);
            this.gpSetupControls.Controls.Add(this.label17);
            this.gpSetupControls.Location = new System.Drawing.Point(296, 4);
            this.gpSetupControls.Name = "gpSetupControls";
            this.gpSetupControls.Size = new System.Drawing.Size(286, 247);
            this.gpSetupControls.TabIndex = 1;
            this.gpSetupControls.TabStop = false;
            this.gpSetupControls.Text = "Special Setup Controls";
            // 
            // label21
            // 
            this.label21.AutoSize = true;
            this.label21.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label21.Location = new System.Drawing.Point(12, 25);
            this.label21.Name = "label21";
            this.label21.Size = new System.Drawing.Size(239, 20);
            this.label21.TabIndex = 47;
            this.label21.Text = "Must be in the \"Setup\" Mode";
            // 
            // KarTechLinearActuatorSetupWidget
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.gpSetupControls);
            this.Controls.Add(this.groupBoxLASetup);
            this.Name = "KarTechLinearActuatorSetupWidget";
            this.Size = new System.Drawing.Size(586, 491);
            this.groupBoxLASetup.ResumeLayout(false);
            this.groupBoxLASetup.PerformLayout();
            this.gpSetupControls.ResumeLayout(false);
            this.gpSetupControls.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBoxLASetup;
        private System.Windows.Forms.TextBox tbMotorMaxPwmPercent;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.TextBox tbMotorMinPwmPercent;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.TextBox tbMotorPWMFreq;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox tbFeedbackLoop_KD;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TextBox tbFeedbackLoop_KI;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox tbFeedbackLoop_KP;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox tbPosReachedErrTime;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox tbMotorMaxCurrentSetting;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbMaxPosition;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbMinPosition;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox tbFeedbackLoopDeadband;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.TextBox tbFeedbackLoopCLFreq;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.Button btnResetConfigs;
        private System.Windows.Forms.Button btnReadConfigParameters;
        private System.Windows.Forms.Button btnSetConfigParameters;
        private System.Windows.Forms.TextBox tbActuatorFunction;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.Button btnAutoZeroPosSensor;
        private System.Windows.Forms.Label label18;
        private System.Windows.Forms.Button btnResetAllCfgs;
        private System.Windows.Forms.Label label16;
        private System.Windows.Forms.Button btnResetOutput;
        private System.Windows.Forms.Label label17;
        private System.Windows.Forms.Button btnSetCmdRptIDs;
        private System.Windows.Forms.Label label20;
        private System.Windows.Forms.Button btnResetHdwrCfgs;
        private System.Windows.Forms.Label label19;
        private System.Windows.Forms.Button btnResetUserConfigs;
        private System.Windows.Forms.GroupBox gpSetupControls;
        private System.Windows.Forms.Label label21;
    }
}
