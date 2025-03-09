namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class CameraMountCorrectionInput
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
            this.gbxCameraMntCorr = new System.Windows.Forms.GroupBox();
            this.tbYawAngleDeg = new System.Windows.Forms.TextBox();
            this.tbPitchAngleDeg = new System.Windows.Forms.TextBox();
            this.tbRollAngleDeg = new System.Windows.Forms.TextBox();
            this.lbYaw = new System.Windows.Forms.Label();
            this.lbPitch = new System.Windows.Forms.Label();
            this.lbRoll = new System.Windows.Forms.Label();
            this.lbDelYPos = new System.Windows.Forms.Label();
            this.lbDelXPos = new System.Windows.Forms.Label();
            this.tbDelYPos = new System.Windows.Forms.TextBox();
            this.tbDelXPos = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.cbMeasUnits = new System.Windows.Forms.ComboBox();
            this.btnSet = new System.Windows.Forms.Button();
            this.btnHelp = new System.Windows.Forms.Button();
            this.gbxCameraMntCorr.SuspendLayout();
            this.SuspendLayout();
            // 
            // gbxCameraMntCorr
            // 
            this.gbxCameraMntCorr.Controls.Add(this.btnHelp);
            this.gbxCameraMntCorr.Controls.Add(this.btnSet);
            this.gbxCameraMntCorr.Controls.Add(this.label1);
            this.gbxCameraMntCorr.Controls.Add(this.cbMeasUnits);
            this.gbxCameraMntCorr.Controls.Add(this.lbDelYPos);
            this.gbxCameraMntCorr.Controls.Add(this.lbDelXPos);
            this.gbxCameraMntCorr.Controls.Add(this.tbDelYPos);
            this.gbxCameraMntCorr.Controls.Add(this.tbDelXPos);
            this.gbxCameraMntCorr.Controls.Add(this.lbRoll);
            this.gbxCameraMntCorr.Controls.Add(this.lbPitch);
            this.gbxCameraMntCorr.Controls.Add(this.lbYaw);
            this.gbxCameraMntCorr.Controls.Add(this.tbRollAngleDeg);
            this.gbxCameraMntCorr.Controls.Add(this.tbPitchAngleDeg);
            this.gbxCameraMntCorr.Controls.Add(this.tbYawAngleDeg);
            this.gbxCameraMntCorr.Location = new System.Drawing.Point(2, 3);
            this.gbxCameraMntCorr.Name = "gbxCameraMntCorr";
            this.gbxCameraMntCorr.Size = new System.Drawing.Size(356, 102);
            this.gbxCameraMntCorr.TabIndex = 0;
            this.gbxCameraMntCorr.TabStop = false;
            this.gbxCameraMntCorr.Text = "Camera Mount Correction";
            // 
            // tbYawAngleDeg
            // 
            this.tbYawAngleDeg.Location = new System.Drawing.Point(68, 19);
            this.tbYawAngleDeg.Name = "tbYawAngleDeg";
            this.tbYawAngleDeg.Size = new System.Drawing.Size(67, 20);
            this.tbYawAngleDeg.TabIndex = 0;
            this.tbYawAngleDeg.Text = "0.0";
            this.tbYawAngleDeg.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // tbPitchAngleDeg
            // 
            this.tbPitchAngleDeg.Location = new System.Drawing.Point(68, 45);
            this.tbPitchAngleDeg.Name = "tbPitchAngleDeg";
            this.tbPitchAngleDeg.Size = new System.Drawing.Size(67, 20);
            this.tbPitchAngleDeg.TabIndex = 1;
            this.tbPitchAngleDeg.Text = "0.0";
            this.tbPitchAngleDeg.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // tbRollAngleDeg
            // 
            this.tbRollAngleDeg.Location = new System.Drawing.Point(68, 71);
            this.tbRollAngleDeg.Name = "tbRollAngleDeg";
            this.tbRollAngleDeg.Size = new System.Drawing.Size(67, 20);
            this.tbRollAngleDeg.TabIndex = 2;
            this.tbRollAngleDeg.Text = "0.0";
            this.tbRollAngleDeg.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // lbYaw
            // 
            this.lbYaw.AutoSize = true;
            this.lbYaw.Location = new System.Drawing.Point(7, 22);
            this.lbYaw.Name = "lbYaw";
            this.lbYaw.Size = new System.Drawing.Size(55, 13);
            this.lbYaw.TabIndex = 3;
            this.lbYaw.Text = "Yaw (deg)";
            // 
            // lbPitch
            // 
            this.lbPitch.AutoSize = true;
            this.lbPitch.Location = new System.Drawing.Point(7, 48);
            this.lbPitch.Name = "lbPitch";
            this.lbPitch.Size = new System.Drawing.Size(58, 13);
            this.lbPitch.TabIndex = 4;
            this.lbPitch.Text = "Pitch (deg)";
            // 
            // lbRoll
            // 
            this.lbRoll.AutoSize = true;
            this.lbRoll.Location = new System.Drawing.Point(7, 74);
            this.lbRoll.Name = "lbRoll";
            this.lbRoll.Size = new System.Drawing.Size(52, 13);
            this.lbRoll.TabIndex = 5;
            this.lbRoll.Text = "Roll (deg)";
            // 
            // lbDelYPos
            // 
            this.lbDelYPos.AutoSize = true;
            this.lbDelYPos.Location = new System.Drawing.Point(150, 48);
            this.lbDelYPos.Name = "lbDelYPos";
            this.lbDelYPos.Size = new System.Drawing.Size(33, 13);
            this.lbDelYPos.TabIndex = 9;
            this.lbDelYPos.Text = "Del Y";
            // 
            // lbDelXPos
            // 
            this.lbDelXPos.AutoSize = true;
            this.lbDelXPos.Location = new System.Drawing.Point(150, 22);
            this.lbDelXPos.Name = "lbDelXPos";
            this.lbDelXPos.Size = new System.Drawing.Size(33, 13);
            this.lbDelXPos.TabIndex = 8;
            this.lbDelXPos.Text = "Del X";
            // 
            // tbDelYPos
            // 
            this.tbDelYPos.Location = new System.Drawing.Point(189, 45);
            this.tbDelYPos.Name = "tbDelYPos";
            this.tbDelYPos.Size = new System.Drawing.Size(67, 20);
            this.tbDelYPos.TabIndex = 7;
            this.tbDelYPos.Text = "0.0";
            this.tbDelYPos.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // tbDelXPos
            // 
            this.tbDelXPos.Location = new System.Drawing.Point(189, 19);
            this.tbDelXPos.Name = "tbDelXPos";
            this.tbDelXPos.Size = new System.Drawing.Size(67, 20);
            this.tbDelXPos.TabIndex = 6;
            this.tbDelXPos.Text = "0.0";
            this.tbDelXPos.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(152, 74);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(31, 13);
            this.label1.TabIndex = 11;
            this.label1.Text = "Units";
            // 
            // cbMeasUnits
            // 
            this.cbMeasUnits.FormattingEnabled = true;
            this.cbMeasUnits.Location = new System.Drawing.Point(189, 70);
            this.cbMeasUnits.Name = "cbMeasUnits";
            this.cbMeasUnits.Size = new System.Drawing.Size(67, 21);
            this.cbMeasUnits.TabIndex = 10;
            this.cbMeasUnits.SelectedIndexChanged += new System.EventHandler(this.cbMeasUnits_SelectedIndexChanged);
            // 
            // btnSet
            // 
            this.btnSet.Location = new System.Drawing.Point(275, 68);
            this.btnSet.Name = "btnSet";
            this.btnSet.Size = new System.Drawing.Size(65, 23);
            this.btnSet.TabIndex = 12;
            this.btnSet.Text = "Set";
            this.btnSet.UseVisualStyleBackColor = true;
            this.btnSet.Click += new System.EventHandler(this.btnSet_Click);
            // 
            // btnHelp
            // 
            this.btnHelp.Location = new System.Drawing.Point(275, 19);
            this.btnHelp.Name = "btnHelp";
            this.btnHelp.Size = new System.Drawing.Size(65, 23);
            this.btnHelp.TabIndex = 13;
            this.btnHelp.Text = "Help";
            this.btnHelp.UseVisualStyleBackColor = true;
            this.btnHelp.Click += new System.EventHandler(this.btnHelp_Click);
            // 
            // CameraMountCorrectionInput
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.gbxCameraMntCorr);
            this.Name = "CameraMountCorrectionInput";
            this.Size = new System.Drawing.Size(358, 100);
            this.gbxCameraMntCorr.ResumeLayout(false);
            this.gbxCameraMntCorr.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox gbxCameraMntCorr;
        private System.Windows.Forms.Label lbDelYPos;
        private System.Windows.Forms.Label lbDelXPos;
        private System.Windows.Forms.TextBox tbDelYPos;
        private System.Windows.Forms.TextBox tbDelXPos;
        private System.Windows.Forms.Label lbRoll;
        private System.Windows.Forms.Label lbPitch;
        private System.Windows.Forms.Label lbYaw;
        private System.Windows.Forms.TextBox tbRollAngleDeg;
        private System.Windows.Forms.TextBox tbPitchAngleDeg;
        private System.Windows.Forms.TextBox tbYawAngleDeg;
        private System.Windows.Forms.Button btnHelp;
        private System.Windows.Forms.Button btnSet;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.ComboBox cbMeasUnits;
    }
}
