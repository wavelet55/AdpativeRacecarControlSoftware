namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class ImageCaptureControl
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
            this.btnSingleImageCapture = new System.Windows.Forms.Button();
            this.btEnableImageCapture = new System.Windows.Forms.Button();
            this.cbContinuousCapture = new System.Windows.Forms.CheckBox();
            this.label2 = new System.Windows.Forms.Label();
            this.nbxNumerOfImgesToCapture = new System.Windows.Forms.NumericUpDown();
            this.groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.nbxNumerOfImgesToCapture)).BeginInit();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.BackColor = System.Drawing.SystemColors.ActiveCaption;
            this.groupBox1.Controls.Add(this.btnSingleImageCapture);
            this.groupBox1.Controls.Add(this.btEnableImageCapture);
            this.groupBox1.Controls.Add(this.cbContinuousCapture);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.nbxNumerOfImgesToCapture);
            this.groupBox1.Location = new System.Drawing.Point(3, 0);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(135, 152);
            this.groupBox1.TabIndex = 2;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Image Capture";
            // 
            // btnSingleImageCapture
            // 
            this.btnSingleImageCapture.Location = new System.Drawing.Point(8, 110);
            this.btnSingleImageCapture.Name = "btnSingleImageCapture";
            this.btnSingleImageCapture.Size = new System.Drawing.Size(114, 23);
            this.btnSingleImageCapture.TabIndex = 6;
            this.btnSingleImageCapture.Text = "Single Step";
            this.btnSingleImageCapture.UseVisualStyleBackColor = true;
            this.btnSingleImageCapture.Click += new System.EventHandler(this.btnSingleImageCapture_Click);
            // 
            // btEnableImageCapture
            // 
            this.btEnableImageCapture.Location = new System.Drawing.Point(8, 81);
            this.btEnableImageCapture.Name = "btEnableImageCapture";
            this.btEnableImageCapture.Size = new System.Drawing.Size(114, 23);
            this.btEnableImageCapture.TabIndex = 5;
            this.btEnableImageCapture.Text = "Enable";
            this.btEnableImageCapture.UseVisualStyleBackColor = true;
            this.btEnableImageCapture.Click += new System.EventHandler(this.btEnableImageCapture_Click);
            // 
            // cbContinuousCapture
            // 
            this.cbContinuousCapture.AutoSize = true;
            this.cbContinuousCapture.Location = new System.Drawing.Point(84, 45);
            this.cbContinuousCapture.Name = "cbContinuousCapture";
            this.cbContinuousCapture.Size = new System.Drawing.Size(51, 17);
            this.cbContinuousCapture.TabIndex = 4;
            this.cbContinuousCapture.Text = "Cont.";
            this.cbContinuousCapture.UseVisualStyleBackColor = true;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(5, 28);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(94, 13);
            this.label2.TabIndex = 3;
            this.label2.Text = "Numer To Capture";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // nbxNumerOfImgesToCapture
            // 
            this.nbxNumerOfImgesToCapture.Location = new System.Drawing.Point(6, 44);
            this.nbxNumerOfImgesToCapture.Maximum = new decimal(new int[] {
            1000000,
            0,
            0,
            0});
            this.nbxNumerOfImgesToCapture.Name = "nbxNumerOfImgesToCapture";
            this.nbxNumerOfImgesToCapture.Size = new System.Drawing.Size(72, 20);
            this.nbxNumerOfImgesToCapture.TabIndex = 2;
            this.nbxNumerOfImgesToCapture.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // ImageCaptureControl
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.groupBox1);
            this.Name = "ImageCaptureControl";
            this.Size = new System.Drawing.Size(141, 155);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.nbxNumerOfImgesToCapture)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.NumericUpDown nbxNumerOfImgesToCapture;
        private System.Windows.Forms.Button btnSingleImageCapture;
        private System.Windows.Forms.Button btEnableImageCapture;
        private System.Windows.Forms.CheckBox cbContinuousCapture;
    }
}
