namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class HeadOrientationSPWidget
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
            this.hScrollBar_HeadTiltLRAngle = new System.Windows.Forms.HScrollBar();
            this.hScrollBar_SipPuffVal = new System.Windows.Forms.HScrollBar();
            this.hScrollBar_HeadRotationLRAngle = new System.Windows.Forms.HScrollBar();
            this.vScrollBar_HeadFrontBackAngle = new System.Windows.Forms.VScrollBar();
            this.tbHeadFrontBackAngle = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.label2 = new System.Windows.Forms.Label();
            this.tbHeadTiltLRAngle = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.tbHeadRotationLRAngle = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.tbSipPuffVal = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.tbSipPuffTotalVal = new System.Windows.Forms.TextBox();
            this.hScrollBar_SipPuffTotalVal = new System.Windows.Forms.HScrollBar();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // hScrollBar_HeadTiltLRAngle
            // 
            this.hScrollBar_HeadTiltLRAngle.Location = new System.Drawing.Point(23, 107);
            this.hScrollBar_HeadTiltLRAngle.Maximum = 45;
            this.hScrollBar_HeadTiltLRAngle.Minimum = -45;
            this.hScrollBar_HeadTiltLRAngle.Name = "hScrollBar_HeadTiltLRAngle";
            this.hScrollBar_HeadTiltLRAngle.Size = new System.Drawing.Size(243, 23);
            this.hScrollBar_HeadTiltLRAngle.TabIndex = 0;
            // 
            // hScrollBar_SipPuffVal
            // 
            this.hScrollBar_SipPuffVal.Location = new System.Drawing.Point(20, 229);
            this.hScrollBar_SipPuffVal.Minimum = -100;
            this.hScrollBar_SipPuffVal.Name = "hScrollBar_SipPuffVal";
            this.hScrollBar_SipPuffVal.Size = new System.Drawing.Size(284, 23);
            this.hScrollBar_SipPuffVal.TabIndex = 1;
            // 
            // hScrollBar_HeadRotationLRAngle
            // 
            this.hScrollBar_HeadRotationLRAngle.Location = new System.Drawing.Point(20, 160);
            this.hScrollBar_HeadRotationLRAngle.Maximum = 90;
            this.hScrollBar_HeadRotationLRAngle.Minimum = -90;
            this.hScrollBar_HeadRotationLRAngle.Name = "hScrollBar_HeadRotationLRAngle";
            this.hScrollBar_HeadRotationLRAngle.Size = new System.Drawing.Size(243, 23);
            this.hScrollBar_HeadRotationLRAngle.TabIndex = 2;
            // 
            // vScrollBar_HeadFrontBackAngle
            // 
            this.vScrollBar_HeadFrontBackAngle.Location = new System.Drawing.Point(282, 17);
            this.vScrollBar_HeadFrontBackAngle.Maximum = 60;
            this.vScrollBar_HeadFrontBackAngle.Minimum = -60;
            this.vScrollBar_HeadFrontBackAngle.Name = "vScrollBar_HeadFrontBackAngle";
            this.vScrollBar_HeadFrontBackAngle.Size = new System.Drawing.Size(25, 169);
            this.vScrollBar_HeadFrontBackAngle.TabIndex = 3;
            // 
            // tbHeadFrontBackAngle
            // 
            this.tbHeadFrontBackAngle.Location = new System.Drawing.Point(199, 45);
            this.tbHeadFrontBackAngle.Name = "tbHeadFrontBackAngle";
            this.tbHeadFrontBackAngle.Size = new System.Drawing.Size(67, 20);
            this.tbHeadFrontBackAngle.TabIndex = 4;
            this.tbHeadFrontBackAngle.Text = "0";
            this.tbHeadFrontBackAngle.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(199, 26);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(61, 13);
            this.label1.TabIndex = 5;
            this.label1.Text = "Front/Back";
            // 
            // groupBox1
            // 
            this.groupBox1.BackColor = System.Drawing.SystemColors.ActiveCaption;
            this.groupBox1.Controls.Add(this.label6);
            this.groupBox1.Controls.Add(this.tbSipPuffTotalVal);
            this.groupBox1.Controls.Add(this.hScrollBar_SipPuffTotalVal);
            this.groupBox1.Controls.Add(this.hScrollBar_HeadRotationLRAngle);
            this.groupBox1.Controls.Add(this.label5);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.tbSipPuffVal);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.hScrollBar_SipPuffVal);
            this.groupBox1.Controls.Add(this.tbHeadRotationLRAngle);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.tbHeadTiltLRAngle);
            this.groupBox1.Location = new System.Drawing.Point(3, 3);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(316, 321);
            this.groupBox1.TabIndex = 6;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Head Orientation";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(78, 84);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(43, 13);
            this.label2.TabIndex = 7;
            this.label2.Text = "Tilt L/R";
            // 
            // tbHeadTiltLRAngle
            // 
            this.tbHeadTiltLRAngle.Location = new System.Drawing.Point(127, 81);
            this.tbHeadTiltLRAngle.Name = "tbHeadTiltLRAngle";
            this.tbHeadTiltLRAngle.Size = new System.Drawing.Size(67, 20);
            this.tbHeadTiltLRAngle.TabIndex = 6;
            this.tbHeadTiltLRAngle.Text = "0";
            this.tbHeadTiltLRAngle.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(52, 140);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(69, 13);
            this.label3.TabIndex = 9;
            this.label3.Text = "Rotation L/R";
            // 
            // tbHeadRotationLRAngle
            // 
            this.tbHeadRotationLRAngle.Location = new System.Drawing.Point(127, 137);
            this.tbHeadRotationLRAngle.Name = "tbHeadRotationLRAngle";
            this.tbHeadRotationLRAngle.Size = new System.Drawing.Size(67, 20);
            this.tbHeadRotationLRAngle.TabIndex = 8;
            this.tbHeadRotationLRAngle.Text = "0";
            this.tbHeadRotationLRAngle.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(38, 209);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(83, 13);
            this.label4.TabIndex = 11;
            this.label4.Text = "Sip And Puff (%)";
            this.label4.Click += new System.EventHandler(this.label4_Click);
            // 
            // tbSipPuffVal
            // 
            this.tbSipPuffVal.Location = new System.Drawing.Point(133, 206);
            this.tbSipPuffVal.Name = "tbSipPuffVal";
            this.tbSipPuffVal.Size = new System.Drawing.Size(67, 20);
            this.tbSipPuffVal.TabIndex = 10;
            this.tbSipPuffVal.Text = "0";
            this.tbSipPuffVal.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label5.Location = new System.Drawing.Point(17, 23);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(120, 13);
            this.label5.TabIndex = 12;
            this.label5.Text = "Orientation Degrees";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(17, 263);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(110, 13);
            this.label6.TabIndex = 15;
            this.label6.Text = "Sip And Puff Total (%)";
            // 
            // tbSipPuffTotalVal
            // 
            this.tbSipPuffTotalVal.Location = new System.Drawing.Point(133, 260);
            this.tbSipPuffTotalVal.Name = "tbSipPuffTotalVal";
            this.tbSipPuffTotalVal.Size = new System.Drawing.Size(67, 20);
            this.tbSipPuffTotalVal.TabIndex = 14;
            this.tbSipPuffTotalVal.Text = "0";
            this.tbSipPuffTotalVal.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // hScrollBar_SipPuffTotalVal
            // 
            this.hScrollBar_SipPuffTotalVal.Location = new System.Drawing.Point(20, 283);
            this.hScrollBar_SipPuffTotalVal.Minimum = -100;
            this.hScrollBar_SipPuffTotalVal.Name = "hScrollBar_SipPuffTotalVal";
            this.hScrollBar_SipPuffTotalVal.Size = new System.Drawing.Size(284, 23);
            this.hScrollBar_SipPuffTotalVal.TabIndex = 13;
            // 
            // HeadOrientationSPWidget
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.label1);
            this.Controls.Add(this.tbHeadFrontBackAngle);
            this.Controls.Add(this.vScrollBar_HeadFrontBackAngle);
            this.Controls.Add(this.hScrollBar_HeadTiltLRAngle);
            this.Controls.Add(this.groupBox1);
            this.Name = "HeadOrientationSPWidget";
            this.Size = new System.Drawing.Size(325, 329);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.HScrollBar hScrollBar_HeadTiltLRAngle;
        private System.Windows.Forms.HScrollBar hScrollBar_SipPuffVal;
        private System.Windows.Forms.HScrollBar hScrollBar_HeadRotationLRAngle;
        private System.Windows.Forms.VScrollBar vScrollBar_HeadFrontBackAngle;
        private System.Windows.Forms.TextBox tbHeadFrontBackAngle;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox tbSipPuffVal;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbHeadRotationLRAngle;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbHeadTiltLRAngle;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox tbSipPuffTotalVal;
        private System.Windows.Forms.HScrollBar hScrollBar_SipPuffTotalVal;
    }
}
