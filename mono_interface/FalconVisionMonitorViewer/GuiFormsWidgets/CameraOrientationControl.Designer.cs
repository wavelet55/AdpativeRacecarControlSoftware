namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class CameraOrientationControl
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
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.btnSend = new System.Windows.Forms.Button();
            this.trackBarAzimuth = new System.Windows.Forms.TrackBar();
            this.trackBarElevation = new System.Windows.Forms.TrackBar();
            this.label2 = new System.Windows.Forms.Label();
            this.tbAzimuth = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.tbCElevation = new System.Windows.Forms.TextBox();
            this.groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarAzimuth)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarElevation)).BeginInit();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.btnSend);
            this.groupBox1.Controls.Add(this.trackBarAzimuth);
            this.groupBox1.Controls.Add(this.trackBarElevation);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.tbAzimuth);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Controls.Add(this.tbCElevation);
            this.groupBox1.Location = new System.Drawing.Point(4, 0);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(294, 173);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Camera Orientation Control";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(17, 102);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(96, 13);
            this.label4.TabIndex = 4;
            this.label4.Text = "(-180 to +180 Deg)";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(17, 40);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(78, 13);
            this.label3.TabIndex = 7;
            this.label3.Text = "(-90 to 90 Deg)";
            // 
            // btnSend
            // 
            this.btnSend.Location = new System.Drawing.Point(201, 18);
            this.btnSend.Name = "btnSend";
            this.btnSend.Size = new System.Drawing.Size(75, 23);
            this.btnSend.TabIndex = 6;
            this.btnSend.Text = "Send";
            this.btnSend.UseVisualStyleBackColor = true;
            this.btnSend.Click += new System.EventHandler(this.btnSend_Click);
            // 
            // trackBarAzimuth
            // 
            this.trackBarAzimuth.Location = new System.Drawing.Point(123, 118);
            this.trackBarAzimuth.Name = "trackBarAzimuth";
            this.trackBarAzimuth.Size = new System.Drawing.Size(153, 45);
            this.trackBarAzimuth.TabIndex = 5;
            this.trackBarAzimuth.Scroll += new System.EventHandler(this.trackBarAzimuth_Scroll);
            // 
            // trackBarElevation
            // 
            this.trackBarElevation.Location = new System.Drawing.Point(123, 56);
            this.trackBarElevation.Name = "trackBarElevation";
            this.trackBarElevation.Size = new System.Drawing.Size(153, 45);
            this.trackBarElevation.TabIndex = 4;
            this.trackBarElevation.Scroll += new System.EventHandler(this.trackBarElevation_Scroll);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(17, 88);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(44, 13);
            this.label2.TabIndex = 3;
            this.label2.Text = "Azimuth";
            // 
            // tbAzimuth
            // 
            this.tbAzimuth.Location = new System.Drawing.Point(17, 118);
            this.tbAzimuth.Name = "tbAzimuth";
            this.tbAzimuth.Size = new System.Drawing.Size(100, 20);
            this.tbAzimuth.TabIndex = 2;
            this.tbAzimuth.Text = "0";
            this.tbAzimuth.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.tbAzimuth.TextChanged += new System.EventHandler(this.tbAzimuth_TextChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(17, 23);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(54, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Elevation ";
            // 
            // tbCElevation
            // 
            this.tbCElevation.Location = new System.Drawing.Point(17, 56);
            this.tbCElevation.Name = "tbCElevation";
            this.tbCElevation.Size = new System.Drawing.Size(100, 20);
            this.tbCElevation.TabIndex = 0;
            this.tbCElevation.Text = "0";
            this.tbCElevation.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.tbCElevation.TextChanged += new System.EventHandler(this.tbCElevation_TextChanged);
            // 
            // CameraOrientationControl
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.groupBox1);
            this.Name = "CameraOrientationControl";
            this.Size = new System.Drawing.Size(304, 177);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarAzimuth)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarElevation)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Button btnSend;
        private System.Windows.Forms.TrackBar trackBarAzimuth;
        private System.Windows.Forms.TrackBar trackBarElevation;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbAzimuth;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox tbCElevation;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label3;
    }
}
