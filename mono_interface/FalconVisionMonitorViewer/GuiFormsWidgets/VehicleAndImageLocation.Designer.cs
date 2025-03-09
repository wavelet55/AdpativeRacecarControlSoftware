namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class VehicleAndImageLocation
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
            this.gbLocationImageInfo = new System.Windows.Forms.GroupBox();
            this.btnFreezeUpdate = new System.Windows.Forms.Button();
            this.tbAltMSL = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.tbImageNumber = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.tbVYaw = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.tbVPitch = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.tbVRoll = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.tbVelNS = new System.Windows.Forms.TextBox();
            this.tbVelEW = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.tbVLon = new System.Windows.Forms.TextBox();
            this.lblLonOrYPos = new System.Windows.Forms.Label();
            this.tbVLat = new System.Windows.Forms.TextBox();
            this.lblLatOrXPos = new System.Windows.Forms.Label();
            this.targetLocation_1 = new FalconVisionMonitorViewer.GuiFormsWidgets.TargetLocation();
            this.targetLocation_2 = new FalconVisionMonitorViewer.GuiFormsWidgets.TargetLocation();
            this.targetLocation_4 = new FalconVisionMonitorViewer.GuiFormsWidgets.TargetLocation();
            this.targetLocation_3 = new FalconVisionMonitorViewer.GuiFormsWidgets.TargetLocation();
            this.cbLatLonXYPos = new System.Windows.Forms.CheckBox();
            this.gbLocationImageInfo.SuspendLayout();
            this.SuspendLayout();
            // 
            // gbLocationImageInfo
            // 
            this.gbLocationImageInfo.Controls.Add(this.cbLatLonXYPos);
            this.gbLocationImageInfo.Controls.Add(this.btnFreezeUpdate);
            this.gbLocationImageInfo.Controls.Add(this.tbAltMSL);
            this.gbLocationImageInfo.Controls.Add(this.label9);
            this.gbLocationImageInfo.Controls.Add(this.label8);
            this.gbLocationImageInfo.Controls.Add(this.tbImageNumber);
            this.gbLocationImageInfo.Controls.Add(this.label7);
            this.gbLocationImageInfo.Controls.Add(this.tbVYaw);
            this.gbLocationImageInfo.Controls.Add(this.label6);
            this.gbLocationImageInfo.Controls.Add(this.tbVPitch);
            this.gbLocationImageInfo.Controls.Add(this.label5);
            this.gbLocationImageInfo.Controls.Add(this.tbVRoll);
            this.gbLocationImageInfo.Controls.Add(this.label3);
            this.gbLocationImageInfo.Controls.Add(this.tbVelNS);
            this.gbLocationImageInfo.Controls.Add(this.tbVelEW);
            this.gbLocationImageInfo.Controls.Add(this.label4);
            this.gbLocationImageInfo.Controls.Add(this.tbVLon);
            this.gbLocationImageInfo.Controls.Add(this.lblLonOrYPos);
            this.gbLocationImageInfo.Controls.Add(this.tbVLat);
            this.gbLocationImageInfo.Controls.Add(this.lblLatOrXPos);
            this.gbLocationImageInfo.Location = new System.Drawing.Point(0, 0);
            this.gbLocationImageInfo.Name = "gbLocationImageInfo";
            this.gbLocationImageInfo.Size = new System.Drawing.Size(597, 114);
            this.gbLocationImageInfo.TabIndex = 0;
            this.gbLocationImageInfo.TabStop = false;
            this.gbLocationImageInfo.Text = "Location and Image Info";
            // 
            // btnFreezeUpdate
            // 
            this.btnFreezeUpdate.Location = new System.Drawing.Point(505, 16);
            this.btnFreezeUpdate.Name = "btnFreezeUpdate";
            this.btnFreezeUpdate.Size = new System.Drawing.Size(75, 23);
            this.btnFreezeUpdate.TabIndex = 19;
            this.btnFreezeUpdate.Text = "Freeze";
            this.btnFreezeUpdate.UseVisualStyleBackColor = true;
            this.btnFreezeUpdate.Click += new System.EventHandler(this.btnFreezeUpdate_Click);
            // 
            // tbAltMSL
            // 
            this.tbAltMSL.Location = new System.Drawing.Point(204, 37);
            this.tbAltMSL.Name = "tbAltMSL";
            this.tbAltMSL.Size = new System.Drawing.Size(71, 20);
            this.tbAltMSL.TabIndex = 18;
            this.tbAltMSL.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(201, 20);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(50, 13);
            this.label9.TabIndex = 17;
            this.label9.Text = "Alt (MSL)";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(355, 21);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(76, 13);
            this.label8.TabIndex = 16;
            this.label8.Text = "Image Number";
            // 
            // tbImageNumber
            // 
            this.tbImageNumber.Location = new System.Drawing.Point(358, 37);
            this.tbImageNumber.Name = "tbImageNumber";
            this.tbImageNumber.Size = new System.Drawing.Size(71, 20);
            this.tbImageNumber.TabIndex = 15;
            this.tbImageNumber.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(355, 66);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(28, 13);
            this.label7.TabIndex = 14;
            this.label7.Text = "Yaw";
            // 
            // tbVYaw
            // 
            this.tbVYaw.Location = new System.Drawing.Point(358, 82);
            this.tbVYaw.Name = "tbVYaw";
            this.tbVYaw.Size = new System.Drawing.Size(71, 20);
            this.tbVYaw.TabIndex = 13;
            this.tbVYaw.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(278, 66);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(31, 13);
            this.label6.TabIndex = 12;
            this.label6.Text = "Pitch";
            // 
            // tbVPitch
            // 
            this.tbVPitch.Location = new System.Drawing.Point(281, 82);
            this.tbVPitch.Name = "tbVPitch";
            this.tbVPitch.Size = new System.Drawing.Size(71, 20);
            this.tbVPitch.TabIndex = 11;
            this.tbVPitch.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(201, 66);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(25, 13);
            this.label5.TabIndex = 10;
            this.label5.Text = "Roll";
            // 
            // tbVRoll
            // 
            this.tbVRoll.Location = new System.Drawing.Point(204, 82);
            this.tbVRoll.Name = "tbVRoll";
            this.tbVRoll.Size = new System.Drawing.Size(71, 20);
            this.tbVRoll.TabIndex = 9;
            this.tbVRoll.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(103, 66);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(73, 13);
            this.label3.TabIndex = 8;
            this.label3.Text = "Vel N/S (M/s)";
            // 
            // tbVelNS
            // 
            this.tbVelNS.Location = new System.Drawing.Point(106, 82);
            this.tbVelNS.Name = "tbVelNS";
            this.tbVelNS.Size = new System.Drawing.Size(71, 20);
            this.tbVelNS.TabIndex = 7;
            this.tbVelNS.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // tbVelEW
            // 
            this.tbVelEW.Location = new System.Drawing.Point(20, 82);
            this.tbVelEW.Name = "tbVelEW";
            this.tbVelEW.Size = new System.Drawing.Size(71, 20);
            this.tbVelEW.TabIndex = 5;
            this.tbVelEW.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(17, 65);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(76, 13);
            this.label4.TabIndex = 4;
            this.label4.Text = "Vel E/W (M/s)";
            // 
            // tbVLon
            // 
            this.tbVLon.Location = new System.Drawing.Point(106, 37);
            this.tbVLon.Name = "tbVLon";
            this.tbVLon.Size = new System.Drawing.Size(71, 20);
            this.tbVLon.TabIndex = 3;
            this.tbVLon.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // lblLonOrYPos
            // 
            this.lblLonOrYPos.AutoSize = true;
            this.lblLonOrYPos.Location = new System.Drawing.Point(103, 20);
            this.lblLonOrYPos.Name = "lblLonOrYPos";
            this.lblLonOrYPos.Size = new System.Drawing.Size(60, 13);
            this.lblLonOrYPos.TabIndex = 2;
            this.lblLonOrYPos.Text = "Y (North m)";
            // 
            // tbVLat
            // 
            this.tbVLat.Location = new System.Drawing.Point(20, 37);
            this.tbVLat.Name = "tbVLat";
            this.tbVLat.Size = new System.Drawing.Size(71, 20);
            this.tbVLat.TabIndex = 1;
            this.tbVLat.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // lblLatOrXPos
            // 
            this.lblLatOrXPos.AutoSize = true;
            this.lblLatOrXPos.Location = new System.Drawing.Point(17, 20);
            this.lblLatOrXPos.Name = "lblLatOrXPos";
            this.lblLatOrXPos.Size = new System.Drawing.Size(55, 13);
            this.lblLatOrXPos.TabIndex = 0;
            this.lblLatOrXPos.Text = "X (East m)";
            // 
            // targetLocation_1
            // 
            this.targetLocation_1.Location = new System.Drawing.Point(-1, 120);
            this.targetLocation_1.Name = "targetLocation_1";
            this.targetLocation_1.Size = new System.Drawing.Size(296, 226);
            this.targetLocation_1.TabIndex = 1;
            // 
            // targetLocation_2
            // 
            this.targetLocation_2.Location = new System.Drawing.Point(301, 120);
            this.targetLocation_2.Name = "targetLocation_2";
            this.targetLocation_2.Size = new System.Drawing.Size(296, 226);
            this.targetLocation_2.TabIndex = 2;
            // 
            // targetLocation_4
            // 
            this.targetLocation_4.Location = new System.Drawing.Point(301, 342);
            this.targetLocation_4.Name = "targetLocation_4";
            this.targetLocation_4.Size = new System.Drawing.Size(296, 226);
            this.targetLocation_4.TabIndex = 4;
            // 
            // targetLocation_3
            // 
            this.targetLocation_3.Location = new System.Drawing.Point(-1, 342);
            this.targetLocation_3.Name = "targetLocation_3";
            this.targetLocation_3.Size = new System.Drawing.Size(296, 226);
            this.targetLocation_3.TabIndex = 3;
            // 
            // cbLatLonXYPos
            // 
            this.cbLatLonXYPos.AutoSize = true;
            this.cbLatLonXYPos.Location = new System.Drawing.Point(452, 84);
            this.cbLatLonXYPos.Name = "cbLatLonXYPos";
            this.cbLatLonXYPos.Size = new System.Drawing.Size(85, 17);
            this.cbLatLonXYPos.TabIndex = 20;
            this.cbLatLonXYPos.Text = "X/Y (meters)";
            this.cbLatLonXYPos.UseVisualStyleBackColor = true;
            this.cbLatLonXYPos.CheckedChanged += new System.EventHandler(this.checkBox1_CheckedChanged);
            // 
            // VehicleAndImageLocation
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.targetLocation_4);
            this.Controls.Add(this.targetLocation_3);
            this.Controls.Add(this.targetLocation_2);
            this.Controls.Add(this.targetLocation_1);
            this.Controls.Add(this.gbLocationImageInfo);
            this.Name = "VehicleAndImageLocation";
            this.Size = new System.Drawing.Size(607, 574);
            this.gbLocationImageInfo.ResumeLayout(false);
            this.gbLocationImageInfo.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox gbLocationImageInfo;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox tbImageNumber;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox tbVYaw;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox tbVPitch;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox tbVRoll;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbVelNS;
        private System.Windows.Forms.TextBox tbVelEW;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox tbVLon;
        private System.Windows.Forms.Label lblLonOrYPos;
        private System.Windows.Forms.TextBox tbVLat;
        private System.Windows.Forms.Label lblLatOrXPos;
        private System.Windows.Forms.Button btnFreezeUpdate;
        private System.Windows.Forms.TextBox tbAltMSL;
        private System.Windows.Forms.Label label9;
        private TargetLocation targetLocation_1;
        private TargetLocation targetLocation_2;
        private TargetLocation targetLocation_4;
        private TargetLocation targetLocation_3;
        private System.Windows.Forms.CheckBox cbLatLonXYPos;
    }
}
