namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class GPSFixWidget
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
            this.tbNoSatelites = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.tbAltitude = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.tbPosX = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.tbPosY = new System.Windows.Forms.TextBox();
            this.cbUnits = new System.Windows.Forms.ComboBox();
            this.label5 = new System.Windows.Forms.Label();
            this.tbSpeed = new System.Windows.Forms.TextBox();
            this.pBarSpeed = new System.Windows.Forms.ProgressBar();
            this.lblSpeedUnits = new System.Windows.Forms.Label();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.lblSpeedUnits);
            this.groupBox1.Controls.Add(this.pBarSpeed);
            this.groupBox1.Controls.Add(this.label5);
            this.groupBox1.Controls.Add(this.tbSpeed);
            this.groupBox1.Controls.Add(this.cbUnits);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.tbPosY);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.tbPosX);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.tbAltitude);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Controls.Add(this.tbNoSatelites);
            this.groupBox1.Location = new System.Drawing.Point(4, 4);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(320, 153);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "GPS Fix";
            // 
            // tbNoSatelites
            // 
            this.tbNoSatelites.Location = new System.Drawing.Point(83, 19);
            this.tbNoSatelites.Name = "tbNoSatelites";
            this.tbNoSatelites.Size = new System.Drawing.Size(32, 20);
            this.tbNoSatelites.TabIndex = 0;
            this.tbNoSatelites.Text = "0";
            this.tbNoSatelites.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(10, 22);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(67, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "No Satelites:";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(206, 22);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(37, 13);
            this.label2.TabIndex = 3;
            this.label2.Text = "Alt (ft):";
            // 
            // tbAltitude
            // 
            this.tbAltitude.Location = new System.Drawing.Point(249, 19);
            this.tbAltitude.Name = "tbAltitude";
            this.tbAltitude.Size = new System.Drawing.Size(58, 20);
            this.tbAltitude.TabIndex = 2;
            this.tbAltitude.Text = "0";
            this.tbAltitude.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(9, 57);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(17, 13);
            this.label3.TabIndex = 5;
            this.label3.Text = "X:";
            // 
            // tbPosX
            // 
            this.tbPosX.Location = new System.Drawing.Point(29, 54);
            this.tbPosX.Name = "tbPosX";
            this.tbPosX.Size = new System.Drawing.Size(82, 20);
            this.tbPosX.TabIndex = 4;
            this.tbPosX.Text = "0";
            this.tbPosX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(117, 57);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(17, 13);
            this.label4.TabIndex = 7;
            this.label4.Text = "Y:";
            // 
            // tbPosY
            // 
            this.tbPosY.Location = new System.Drawing.Point(137, 54);
            this.tbPosY.Name = "tbPosY";
            this.tbPosY.Size = new System.Drawing.Size(80, 20);
            this.tbPosY.TabIndex = 6;
            this.tbPosY.Text = "0";
            this.tbPosY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // cbUnits
            // 
            this.cbUnits.FormattingEnabled = true;
            this.cbUnits.Location = new System.Drawing.Point(239, 53);
            this.cbUnits.Name = "cbUnits";
            this.cbUnits.Size = new System.Drawing.Size(68, 21);
            this.cbUnits.TabIndex = 8;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(88, 93);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(41, 13);
            this.label5.TabIndex = 10;
            this.label5.Text = "Speed:";
            // 
            // tbSpeed
            // 
            this.tbSpeed.Location = new System.Drawing.Point(135, 90);
            this.tbSpeed.Name = "tbSpeed";
            this.tbSpeed.Size = new System.Drawing.Size(64, 20);
            this.tbSpeed.TabIndex = 9;
            this.tbSpeed.Text = "0";
            this.tbSpeed.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // pBarSpeed
            // 
            this.pBarSpeed.Location = new System.Drawing.Point(12, 121);
            this.pBarSpeed.Name = "pBarSpeed";
            this.pBarSpeed.Size = new System.Drawing.Size(295, 23);
            this.pBarSpeed.TabIndex = 11;
            // 
            // lblSpeedUnits
            // 
            this.lblSpeedUnits.AutoSize = true;
            this.lblSpeedUnits.Location = new System.Drawing.Point(206, 93);
            this.lblSpeedUnits.Name = "lblSpeedUnits";
            this.lblSpeedUnits.Size = new System.Drawing.Size(31, 13);
            this.lblSpeedUnits.TabIndex = 12;
            this.lblSpeedUnits.Text = "MPH";
            // 
            // GPSFixWidget
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.groupBox1);
            this.Name = "GPSFixWidget";
            this.Size = new System.Drawing.Size(327, 160);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.ProgressBar pBarSpeed;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox tbSpeed;
        private System.Windows.Forms.ComboBox cbUnits;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox tbPosY;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbPosX;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbAltitude;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox tbNoSatelites;
        private System.Windows.Forms.Label lblSpeedUnits;
    }
}
