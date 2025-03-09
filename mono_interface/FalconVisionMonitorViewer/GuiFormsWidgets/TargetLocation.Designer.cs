namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class TargetLocation
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
            this.gboxTargetInfo = new System.Windows.Forms.GroupBox();
            this.label7 = new System.Windows.Forms.Label();
            this.tbTgtElevation = new System.Windows.Forms.TextBox();
            this.tbTgtAzimuth = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.tbTgtOrientationAngle = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.tbCv11 = new System.Windows.Forms.TextBox();
            this.tbCv10 = new System.Windows.Forms.TextBox();
            this.tbCv01 = new System.Windows.Forms.TextBox();
            this.tbCv00 = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.btTargetType = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.tbTgtPxlY = new System.Windows.Forms.TextBox();
            this.tbTgtPxlX = new System.Windows.Forms.TextBox();
            this.lblLatLonXY = new System.Windows.Forms.Label();
            this.tbTgtLon = new System.Windows.Forms.TextBox();
            this.tbTgtLat = new System.Windows.Forms.TextBox();
            this.cbLatLonXYDisp = new System.Windows.Forms.CheckBox();
            this.gboxTargetInfo.SuspendLayout();
            this.SuspendLayout();
            // 
            // gboxTargetInfo
            // 
            this.gboxTargetInfo.Controls.Add(this.cbLatLonXYDisp);
            this.gboxTargetInfo.Controls.Add(this.label7);
            this.gboxTargetInfo.Controls.Add(this.tbTgtElevation);
            this.gboxTargetInfo.Controls.Add(this.tbTgtAzimuth);
            this.gboxTargetInfo.Controls.Add(this.label6);
            this.gboxTargetInfo.Controls.Add(this.tbTgtOrientationAngle);
            this.gboxTargetInfo.Controls.Add(this.label5);
            this.gboxTargetInfo.Controls.Add(this.tbCv11);
            this.gboxTargetInfo.Controls.Add(this.tbCv10);
            this.gboxTargetInfo.Controls.Add(this.tbCv01);
            this.gboxTargetInfo.Controls.Add(this.tbCv00);
            this.gboxTargetInfo.Controls.Add(this.label4);
            this.gboxTargetInfo.Controls.Add(this.btTargetType);
            this.gboxTargetInfo.Controls.Add(this.label3);
            this.gboxTargetInfo.Controls.Add(this.tbTgtPxlY);
            this.gboxTargetInfo.Controls.Add(this.tbTgtPxlX);
            this.gboxTargetInfo.Controls.Add(this.lblLatLonXY);
            this.gboxTargetInfo.Controls.Add(this.tbTgtLon);
            this.gboxTargetInfo.Controls.Add(this.tbTgtLat);
            this.gboxTargetInfo.Location = new System.Drawing.Point(4, 4);
            this.gboxTargetInfo.Name = "gboxTargetInfo";
            this.gboxTargetInfo.Size = new System.Drawing.Size(289, 217);
            this.gboxTargetInfo.TabIndex = 0;
            this.gboxTargetInfo.TabStop = false;
            this.gboxTargetInfo.Text = "Target Number: ";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(6, 106);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(48, 13);
            this.label7.TabIndex = 19;
            this.label7.Text = "Az/Elev:";
            // 
            // tbTgtElevation
            // 
            this.tbTgtElevation.Location = new System.Drawing.Point(170, 103);
            this.tbTgtElevation.Name = "tbTgtElevation";
            this.tbTgtElevation.Size = new System.Drawing.Size(100, 20);
            this.tbTgtElevation.TabIndex = 18;
            this.tbTgtElevation.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // tbTgtAzimuth
            // 
            this.tbTgtAzimuth.Location = new System.Drawing.Point(64, 103);
            this.tbTgtAzimuth.Name = "tbTgtAzimuth";
            this.tbTgtAzimuth.Size = new System.Drawing.Size(100, 20);
            this.tbTgtAzimuth.TabIndex = 17;
            this.tbTgtAzimuth.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(22, 165);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(88, 13);
            this.label6.TabIndex = 16;
            this.label6.Text = "Orientation Angle";
            // 
            // tbTgtOrientationAngle
            // 
            this.tbTgtOrientationAngle.Location = new System.Drawing.Point(25, 181);
            this.tbTgtOrientationAngle.Name = "tbTgtOrientationAngle";
            this.tbTgtOrientationAngle.Size = new System.Drawing.Size(85, 20);
            this.tbTgtOrientationAngle.TabIndex = 16;
            this.tbTgtOrientationAngle.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(156, 139);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(92, 13);
            this.label5.TabIndex = 15;
            this.label5.Text = "Covarience Matrix";
            // 
            // tbCv11
            // 
            this.tbCv11.Location = new System.Drawing.Point(208, 181);
            this.tbCv11.Name = "tbCv11";
            this.tbCv11.Size = new System.Drawing.Size(62, 20);
            this.tbCv11.TabIndex = 14;
            this.tbCv11.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // tbCv10
            // 
            this.tbCv10.Location = new System.Drawing.Point(140, 181);
            this.tbCv10.Name = "tbCv10";
            this.tbCv10.Size = new System.Drawing.Size(62, 20);
            this.tbCv10.TabIndex = 13;
            this.tbCv10.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // tbCv01
            // 
            this.tbCv01.Location = new System.Drawing.Point(208, 155);
            this.tbCv01.Name = "tbCv01";
            this.tbCv01.Size = new System.Drawing.Size(62, 20);
            this.tbCv01.TabIndex = 12;
            this.tbCv01.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // tbCv00
            // 
            this.tbCv00.Location = new System.Drawing.Point(140, 155);
            this.tbCv00.Name = "tbCv00";
            this.tbCv00.Size = new System.Drawing.Size(62, 20);
            this.tbCv00.TabIndex = 11;
            this.tbCv00.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(99, 27);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(68, 13);
            this.label4.TabIndex = 10;
            this.label4.Text = "Target Type:";
            // 
            // btTargetType
            // 
            this.btTargetType.Location = new System.Drawing.Point(170, 22);
            this.btTargetType.Name = "btTargetType";
            this.btTargetType.Size = new System.Drawing.Size(100, 23);
            this.btTargetType.TabIndex = 9;
            this.btTargetType.Text = "No Target";
            this.btTargetType.UseMnemonic = false;
            this.btTargetType.UseVisualStyleBackColor = true;
            this.btTargetType.Click += new System.EventHandler(this.btTargetType_Click);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(6, 80);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(32, 13);
            this.label3.TabIndex = 8;
            this.label3.Text = "Pixel:";
            // 
            // tbTgtPxlY
            // 
            this.tbTgtPxlY.Location = new System.Drawing.Point(170, 77);
            this.tbTgtPxlY.Name = "tbTgtPxlY";
            this.tbTgtPxlY.Size = new System.Drawing.Size(100, 20);
            this.tbTgtPxlY.TabIndex = 7;
            this.tbTgtPxlY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // tbTgtPxlX
            // 
            this.tbTgtPxlX.Location = new System.Drawing.Point(64, 77);
            this.tbTgtPxlX.Name = "tbTgtPxlX";
            this.tbTgtPxlX.Size = new System.Drawing.Size(100, 20);
            this.tbTgtPxlX.TabIndex = 6;
            this.tbTgtPxlX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // lblLatLonXY
            // 
            this.lblLatLonXY.AutoSize = true;
            this.lblLatLonXY.Location = new System.Drawing.Point(6, 54);
            this.lblLatLonXY.Name = "lblLatLonXY";
            this.lblLatLonXY.Size = new System.Drawing.Size(43, 13);
            this.lblLatLonXY.TabIndex = 2;
            this.lblLatLonXY.Text = "X/Y (m)";
            // 
            // tbTgtLon
            // 
            this.tbTgtLon.Location = new System.Drawing.Point(170, 51);
            this.tbTgtLon.Name = "tbTgtLon";
            this.tbTgtLon.Size = new System.Drawing.Size(100, 20);
            this.tbTgtLon.TabIndex = 1;
            this.tbTgtLon.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // tbTgtLat
            // 
            this.tbTgtLat.Location = new System.Drawing.Point(64, 51);
            this.tbTgtLat.Name = "tbTgtLat";
            this.tbTgtLat.Size = new System.Drawing.Size(100, 20);
            this.tbTgtLat.TabIndex = 0;
            this.tbTgtLat.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // cbLatLonXYDisp
            // 
            this.cbLatLonXYDisp.AutoSize = true;
            this.cbLatLonXYDisp.Location = new System.Drawing.Point(9, 27);
            this.cbLatLonXYDisp.Name = "cbLatLonXYDisp";
            this.cbLatLonXYDisp.Size = new System.Drawing.Size(51, 17);
            this.cbLatLonXYDisp.TabIndex = 20;
            this.cbLatLonXYDisp.Text = "X / Y";
            this.cbLatLonXYDisp.UseVisualStyleBackColor = true;
            this.cbLatLonXYDisp.CheckedChanged += new System.EventHandler(this.cbLatLonXYDisp_CheckedChanged);
            // 
            // TargetLocation
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.gboxTargetInfo);
            this.Name = "TargetLocation";
            this.Size = new System.Drawing.Size(296, 226);
            this.gboxTargetInfo.ResumeLayout(false);
            this.gboxTargetInfo.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox gboxTargetInfo;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbTgtPxlY;
        private System.Windows.Forms.TextBox tbTgtPxlX;
        private System.Windows.Forms.Label lblLatLonXY;
        private System.Windows.Forms.TextBox tbTgtLon;
        private System.Windows.Forms.TextBox tbTgtLat;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Button btTargetType;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox tbTgtOrientationAngle;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox tbCv11;
        private System.Windows.Forms.TextBox tbCv10;
        private System.Windows.Forms.TextBox tbCv01;
        private System.Windows.Forms.TextBox tbCv00;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox tbTgtElevation;
        private System.Windows.Forms.TextBox tbTgtAzimuth;
        private System.Windows.Forms.CheckBox cbLatLonXYDisp;
    }
}
