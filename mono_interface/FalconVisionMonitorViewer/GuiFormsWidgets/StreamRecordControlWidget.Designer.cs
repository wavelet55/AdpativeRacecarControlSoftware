namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class StreamRecordControlWidget
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
            this.gbStreamRecordControl = new System.Windows.Forms.GroupBox();
            this.btnRetrieveMsg = new System.Windows.Forms.Button();
            this.btnSendMsg = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.tbCompressedImgQuality = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.tbStreamFPS = new System.Windows.Forms.TextBox();
            this.cbCompressRecordImageEnable = new System.Windows.Forms.CheckBox();
            this.cbRecordEnable = new System.Windows.Forms.CheckBox();
            this.cbStreamEnable = new System.Windows.Forms.CheckBox();
            this.label3 = new System.Windows.Forms.Label();
            this.tbImageScaleDownFactor = new System.Windows.Forms.TextBox();
            this.gbStreamRecordControl.SuspendLayout();
            this.SuspendLayout();
            // 
            // gbStreamRecordControl
            // 
            this.gbStreamRecordControl.Controls.Add(this.label3);
            this.gbStreamRecordControl.Controls.Add(this.tbImageScaleDownFactor);
            this.gbStreamRecordControl.Controls.Add(this.btnRetrieveMsg);
            this.gbStreamRecordControl.Controls.Add(this.btnSendMsg);
            this.gbStreamRecordControl.Controls.Add(this.label2);
            this.gbStreamRecordControl.Controls.Add(this.tbCompressedImgQuality);
            this.gbStreamRecordControl.Controls.Add(this.label1);
            this.gbStreamRecordControl.Controls.Add(this.tbStreamFPS);
            this.gbStreamRecordControl.Controls.Add(this.cbCompressRecordImageEnable);
            this.gbStreamRecordControl.Controls.Add(this.cbRecordEnable);
            this.gbStreamRecordControl.Controls.Add(this.cbStreamEnable);
            this.gbStreamRecordControl.Location = new System.Drawing.Point(4, 4);
            this.gbStreamRecordControl.Name = "gbStreamRecordControl";
            this.gbStreamRecordControl.Size = new System.Drawing.Size(233, 153);
            this.gbStreamRecordControl.TabIndex = 0;
            this.gbStreamRecordControl.TabStop = false;
            this.gbStreamRecordControl.Text = "Stream and Record Control";
            // 
            // btnRetrieveMsg
            // 
            this.btnRetrieveMsg.Location = new System.Drawing.Point(164, 96);
            this.btnRetrieveMsg.Name = "btnRetrieveMsg";
            this.btnRetrieveMsg.Size = new System.Drawing.Size(57, 23);
            this.btnRetrieveMsg.TabIndex = 8;
            this.btnRetrieveMsg.Text = "Retrieve";
            this.btnRetrieveMsg.UseVisualStyleBackColor = true;
            this.btnRetrieveMsg.Click += new System.EventHandler(this.btnRetrieveMsg_Click);
            // 
            // btnSendMsg
            // 
            this.btnSendMsg.Location = new System.Drawing.Point(164, 37);
            this.btnSendMsg.Name = "btnSendMsg";
            this.btnSendMsg.Size = new System.Drawing.Size(57, 23);
            this.btnSendMsg.TabIndex = 7;
            this.btnSendMsg.Text = "Send";
            this.btnSendMsg.UseVisualStyleBackColor = true;
            this.btnSendMsg.Click += new System.EventHandler(this.btnSendMsg_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(58, 108);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(94, 13);
            this.label2.TabIndex = 6;
            this.label2.Text = "Compress Img Qlty";
            // 
            // tbCompressedImgQuality
            // 
            this.tbCompressedImgQuality.Location = new System.Drawing.Point(6, 105);
            this.tbCompressedImgQuality.Name = "tbCompressedImgQuality";
            this.tbCompressedImgQuality.Size = new System.Drawing.Size(46, 20);
            this.tbCompressedImgQuality.TabIndex = 5;
            this.tbCompressedImgQuality.Text = "50";
            this.tbCompressedImgQuality.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(58, 86);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(63, 13);
            this.label1.TabIndex = 4;
            this.label1.Text = "Stream FPS";
            // 
            // tbStreamFPS
            // 
            this.tbStreamFPS.Location = new System.Drawing.Point(6, 83);
            this.tbStreamFPS.Name = "tbStreamFPS";
            this.tbStreamFPS.Size = new System.Drawing.Size(46, 20);
            this.tbStreamFPS.TabIndex = 3;
            this.tbStreamFPS.Text = "5.0";
            this.tbStreamFPS.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // cbCompressRecordImageEnable
            // 
            this.cbCompressRecordImageEnable.AutoSize = true;
            this.cbCompressRecordImageEnable.Location = new System.Drawing.Point(6, 60);
            this.cbCompressRecordImageEnable.Name = "cbCompressRecordImageEnable";
            this.cbCompressRecordImageEnable.Size = new System.Drawing.Size(142, 17);
            this.cbCompressRecordImageEnable.TabIndex = 2;
            this.cbCompressRecordImageEnable.Text = "Compress Record Image";
            this.cbCompressRecordImageEnable.UseVisualStyleBackColor = true;
            // 
            // cbRecordEnable
            // 
            this.cbRecordEnable.AutoSize = true;
            this.cbRecordEnable.Location = new System.Drawing.Point(7, 37);
            this.cbRecordEnable.Name = "cbRecordEnable";
            this.cbRecordEnable.Size = new System.Drawing.Size(97, 17);
            this.cbRecordEnable.TabIndex = 1;
            this.cbRecordEnable.Text = "Record Enable";
            this.cbRecordEnable.UseVisualStyleBackColor = true;
            // 
            // cbStreamEnable
            // 
            this.cbStreamEnable.AutoSize = true;
            this.cbStreamEnable.Location = new System.Drawing.Point(7, 20);
            this.cbStreamEnable.Name = "cbStreamEnable";
            this.cbStreamEnable.Size = new System.Drawing.Size(95, 17);
            this.cbStreamEnable.TabIndex = 0;
            this.cbStreamEnable.Text = "Stream Enable";
            this.cbStreamEnable.UseVisualStyleBackColor = true;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(58, 130);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(98, 13);
            this.label3.TabIndex = 10;
            this.label3.Text = "Scale Down Factor";
            // 
            // tbImageScaleDownFactor
            // 
            this.tbImageScaleDownFactor.Location = new System.Drawing.Point(6, 127);
            this.tbImageScaleDownFactor.Name = "tbImageScaleDownFactor";
            this.tbImageScaleDownFactor.Size = new System.Drawing.Size(46, 20);
            this.tbImageScaleDownFactor.TabIndex = 9;
            this.tbImageScaleDownFactor.Text = "2.5";
            this.tbImageScaleDownFactor.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // StreamRecordControlWidget
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.gbStreamRecordControl);
            this.Name = "StreamRecordControlWidget";
            this.Size = new System.Drawing.Size(242, 160);
            this.gbStreamRecordControl.ResumeLayout(false);
            this.gbStreamRecordControl.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox gbStreamRecordControl;
        private System.Windows.Forms.CheckBox cbCompressRecordImageEnable;
        private System.Windows.Forms.CheckBox cbRecordEnable;
        private System.Windows.Forms.CheckBox cbStreamEnable;
        private System.Windows.Forms.Button btnRetrieveMsg;
        private System.Windows.Forms.Button btnSendMsg;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbCompressedImgQuality;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox tbStreamFPS;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbImageScaleDownFactor;
    }
}
