namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class HeadOrientationCalWidget
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
            this.label1 = new System.Windows.Forms.Label();
            this.tbCalState = new System.Windows.Forms.TextBox();
            this.tbCalMessage = new System.Windows.Forms.TextBox();
            this.btnCalStart = new System.Windows.Forms.Button();
            this.btnRejectImage = new System.Windows.Forms.Button();
            this.btnImageOK = new System.Windows.Forms.Button();
            this.btnCaptureImage = new System.Windows.Forms.Button();
            this.tbImageNumber = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.tbImageNumber);
            this.groupBox1.Controls.Add(this.btnRejectImage);
            this.groupBox1.Controls.Add(this.btnImageOK);
            this.groupBox1.Controls.Add(this.btnCaptureImage);
            this.groupBox1.Controls.Add(this.btnCalStart);
            this.groupBox1.Controls.Add(this.tbCalMessage);
            this.groupBox1.Controls.Add(this.tbCalState);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Location = new System.Drawing.Point(4, 4);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(250, 164);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Head Orientation Calibration";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(7, 20);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(50, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Cal State";
            // 
            // tbCalState
            // 
            this.tbCalState.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tbCalState.Location = new System.Drawing.Point(10, 37);
            this.tbCalState.Name = "tbCalState";
            this.tbCalState.ReadOnly = true;
            this.tbCalState.Size = new System.Drawing.Size(133, 20);
            this.tbCalState.TabIndex = 1;
            this.tbCalState.Text = "Unknown";
            // 
            // tbCalMessage
            // 
            this.tbCalMessage.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tbCalMessage.Location = new System.Drawing.Point(10, 63);
            this.tbCalMessage.Name = "tbCalMessage";
            this.tbCalMessage.ReadOnly = true;
            this.tbCalMessage.Size = new System.Drawing.Size(234, 20);
            this.tbCalMessage.TabIndex = 3;
            this.tbCalMessage.Text = "Cal Message";
            // 
            // btnCalStart
            // 
            this.btnCalStart.Location = new System.Drawing.Point(169, 20);
            this.btnCalStart.Name = "btnCalStart";
            this.btnCalStart.Size = new System.Drawing.Size(75, 23);
            this.btnCalStart.TabIndex = 4;
            this.btnCalStart.Text = "Start";
            this.btnCalStart.UseVisualStyleBackColor = true;
            this.btnCalStart.Click += new System.EventHandler(this.btnCalStart_Click);
            // 
            // btnRejectImage
            // 
            this.btnRejectImage.Location = new System.Drawing.Point(169, 127);
            this.btnRejectImage.Name = "btnRejectImage";
            this.btnRejectImage.Size = new System.Drawing.Size(75, 23);
            this.btnRejectImage.TabIndex = 14;
            this.btnRejectImage.Text = "Reject Img";
            this.btnRejectImage.UseVisualStyleBackColor = true;
            this.btnRejectImage.Click += new System.EventHandler(this.btnRejectImage_Click);
            // 
            // btnImageOK
            // 
            this.btnImageOK.Location = new System.Drawing.Point(169, 98);
            this.btnImageOK.Name = "btnImageOK";
            this.btnImageOK.Size = new System.Drawing.Size(75, 23);
            this.btnImageOK.TabIndex = 13;
            this.btnImageOK.Text = "Image Ok";
            this.btnImageOK.UseVisualStyleBackColor = true;
            this.btnImageOK.Click += new System.EventHandler(this.btnImageOK_Click);
            // 
            // btnCaptureImage
            // 
            this.btnCaptureImage.Location = new System.Drawing.Point(78, 112);
            this.btnCaptureImage.Name = "btnCaptureImage";
            this.btnCaptureImage.Size = new System.Drawing.Size(75, 23);
            this.btnCaptureImage.TabIndex = 12;
            this.btnCaptureImage.Text = "Capture Img";
            this.btnCaptureImage.UseVisualStyleBackColor = true;
            this.btnCaptureImage.Click += new System.EventHandler(this.btnCaptureImage_Click);
            // 
            // tbImageNumber
            // 
            this.tbImageNumber.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tbImageNumber.Location = new System.Drawing.Point(10, 112);
            this.tbImageNumber.Name = "tbImageNumber";
            this.tbImageNumber.ReadOnly = true;
            this.tbImageNumber.Size = new System.Drawing.Size(47, 20);
            this.tbImageNumber.TabIndex = 15;
            this.tbImageNumber.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(7, 98);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(53, 13);
            this.label2.TabIndex = 16;
            this.label2.Text = "Image No";
            // 
            // HeadOrientationCalWidget
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.groupBox1);
            this.Name = "HeadOrientationCalWidget";
            this.Size = new System.Drawing.Size(260, 171);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Button btnCalStart;
        private System.Windows.Forms.TextBox tbCalMessage;
        private System.Windows.Forms.TextBox tbCalState;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button btnRejectImage;
        private System.Windows.Forms.Button btnImageOK;
        private System.Windows.Forms.Button btnCaptureImage;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbImageNumber;
    }
}
