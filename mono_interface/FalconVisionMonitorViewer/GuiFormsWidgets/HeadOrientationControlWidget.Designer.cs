namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class HeadOrientationControlWidget
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
            this.cbxOrientationTypeOutpSelect = new System.Windows.Forms.ComboBox();
            this.label1 = new System.Windows.Forms.Label();
            this.ckbDisableHeadKalmanFilter = new System.Windows.Forms.CheckBox();
            this.ckbDisableCarOrInp = new System.Windows.Forms.CheckBox();
            this.ckbDisableCarGravityFB = new System.Windows.Forms.CheckBox();
            this.tbHeadQvar = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.tbHeadRvar = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.tbGravityFBGain = new System.Windows.Forms.TextBox();
            this.btnSend = new System.Windows.Forms.Button();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.SuspendLayout();
            // 
            // cbxOrientationTypeOutpSelect
            // 
            this.cbxOrientationTypeOutpSelect.FormattingEnabled = true;
            this.cbxOrientationTypeOutpSelect.Location = new System.Drawing.Point(23, 42);
            this.cbxOrientationTypeOutpSelect.Name = "cbxOrientationTypeOutpSelect";
            this.cbxOrientationTypeOutpSelect.Size = new System.Drawing.Size(130, 21);
            this.cbxOrientationTypeOutpSelect.TabIndex = 0;
            this.cbxOrientationTypeOutpSelect.SelectedIndexChanged += new System.EventHandler(this.cbxOrientationTypeOutpSelect_SelectedIndexChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(23, 21);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(120, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Orientation Type Output";
            // 
            // ckbDisableHeadKalmanFilter
            // 
            this.ckbDisableHeadKalmanFilter.AutoSize = true;
            this.ckbDisableHeadKalmanFilter.Location = new System.Drawing.Point(174, 21);
            this.ckbDisableHeadKalmanFilter.Name = "ckbDisableHeadKalmanFilter";
            this.ckbDisableHeadKalmanFilter.Size = new System.Drawing.Size(153, 17);
            this.ckbDisableHeadKalmanFilter.TabIndex = 2;
            this.ckbDisableHeadKalmanFilter.Text = "Disable Head Kalman Filter";
            this.ckbDisableHeadKalmanFilter.UseVisualStyleBackColor = true;
            this.ckbDisableHeadKalmanFilter.CheckedChanged += new System.EventHandler(this.ckbDisableHeadKalmanFilter_CheckedChanged);
            // 
            // ckbDisableCarOrInp
            // 
            this.ckbDisableCarOrInp.AutoSize = true;
            this.ckbDisableCarOrInp.Location = new System.Drawing.Point(174, 46);
            this.ckbDisableCarOrInp.Name = "ckbDisableCarOrInp";
            this.ckbDisableCarOrInp.Size = new System.Drawing.Size(152, 17);
            this.ckbDisableCarOrInp.TabIndex = 3;
            this.ckbDisableCarOrInp.Text = "Disable Car Orientation Inp";
            this.ckbDisableCarOrInp.UseVisualStyleBackColor = true;
            this.ckbDisableCarOrInp.CheckedChanged += new System.EventHandler(this.ckbDisableCarOrInp_CheckedChanged);
            // 
            // ckbDisableCarGravityFB
            // 
            this.ckbDisableCarGravityFB.AutoSize = true;
            this.ckbDisableCarGravityFB.Location = new System.Drawing.Point(174, 69);
            this.ckbDisableCarGravityFB.Name = "ckbDisableCarGravityFB";
            this.ckbDisableCarGravityFB.Size = new System.Drawing.Size(132, 17);
            this.ckbDisableCarGravityFB.TabIndex = 4;
            this.ckbDisableCarGravityFB.Text = "Disable Car Gravity FB";
            this.ckbDisableCarGravityFB.UseVisualStyleBackColor = true;
            this.ckbDisableCarGravityFB.CheckedChanged += new System.EventHandler(this.ckbDisableCarGravityFB_CheckedChanged);
            // 
            // tbHeadQvar
            // 
            this.tbHeadQvar.Location = new System.Drawing.Point(342, 42);
            this.tbHeadQvar.Name = "tbHeadQvar";
            this.tbHeadQvar.Size = new System.Drawing.Size(72, 20);
            this.tbHeadQvar.TabIndex = 5;
            this.tbHeadQvar.Text = "0.001";
            this.tbHeadQvar.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(339, 25);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(59, 13);
            this.label2.TabIndex = 6;
            this.label2.Text = "Head Qvar";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(426, 25);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(59, 13);
            this.label3.TabIndex = 8;
            this.label3.Text = "Head Rvar";
            // 
            // tbHeadRvar
            // 
            this.tbHeadRvar.Location = new System.Drawing.Point(429, 42);
            this.tbHeadRvar.Name = "tbHeadRvar";
            this.tbHeadRvar.Size = new System.Drawing.Size(70, 20);
            this.tbHeadRvar.TabIndex = 7;
            this.tbHeadRvar.Text = "0.001";
            this.tbHeadRvar.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(323, 73);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(100, 13);
            this.label4.TabIndex = 10;
            this.label4.Text = "Car Gravity FB Gain";
            // 
            // tbGravityFBGain
            // 
            this.tbGravityFBGain.Location = new System.Drawing.Point(429, 69);
            this.tbGravityFBGain.Name = "tbGravityFBGain";
            this.tbGravityFBGain.Size = new System.Drawing.Size(70, 20);
            this.tbGravityFBGain.TabIndex = 9;
            this.tbGravityFBGain.Text = "0.99";
            this.tbGravityFBGain.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // btnSend
            // 
            this.btnSend.Location = new System.Drawing.Point(514, 66);
            this.btnSend.Name = "btnSend";
            this.btnSend.Size = new System.Drawing.Size(75, 23);
            this.btnSend.TabIndex = 11;
            this.btnSend.Text = "Send";
            this.btnSend.UseVisualStyleBackColor = true;
            this.btnSend.Click += new System.EventHandler(this.btnSend_Click);
            // 
            // groupBox1
            // 
            this.groupBox1.Location = new System.Drawing.Point(4, 4);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(601, 100);
            this.groupBox1.TabIndex = 12;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Head Orientation Control";
            // 
            // HeadOrientationControlWidget
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.btnSend);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.tbGravityFBGain);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.tbHeadRvar);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.tbHeadQvar);
            this.Controls.Add(this.ckbDisableCarGravityFB);
            this.Controls.Add(this.ckbDisableCarOrInp);
            this.Controls.Add(this.ckbDisableHeadKalmanFilter);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.cbxOrientationTypeOutpSelect);
            this.Controls.Add(this.groupBox1);
            this.Name = "HeadOrientationControlWidget";
            this.Size = new System.Drawing.Size(609, 107);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ComboBox cbxOrientationTypeOutpSelect;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.CheckBox ckbDisableHeadKalmanFilter;
        private System.Windows.Forms.CheckBox ckbDisableCarOrInp;
        private System.Windows.Forms.CheckBox ckbDisableCarGravityFB;
        private System.Windows.Forms.TextBox tbHeadQvar;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbHeadRvar;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox tbGravityFBGain;
        private System.Windows.Forms.Button btnSend;
        private System.Windows.Forms.GroupBox groupBox1;
    }
}
