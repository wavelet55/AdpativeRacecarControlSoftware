namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class HeadTrackingControlWidget
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
            this.gBoxHeadTrackingCtrl = new System.Windows.Forms.GroupBox();
            this.btnSend = new System.Windows.Forms.Button();
            this.label8 = new System.Windows.Forms.Label();
            this.cBoxDisplayType = new System.Windows.Forms.ComboBox();
            this.label7 = new System.Windows.Forms.Label();
            this.tbhSBConfidencePercent = new System.Windows.Forms.TextBox();
            this.hSBConfidencePercent = new System.Windows.Forms.HScrollBar();
            this.label5 = new System.Windows.Forms.Label();
            this.tbReprojectionError = new System.Windows.Forms.TextBox();
            this.hSBReprojectionError = new System.Windows.Forms.HScrollBar();
            this.label6 = new System.Windows.Forms.Label();
            this.tbNoIterations = new System.Windows.Forms.TextBox();
            this.hSBNoIterations = new System.Windows.Forms.HScrollBar();
            this.label3 = new System.Windows.Forms.Label();
            this.tbGlyphAreaMax = new System.Windows.Forms.TextBox();
            this.hSBGlyphAreaMax = new System.Windows.Forms.HScrollBar();
            this.label4 = new System.Windows.Forms.Label();
            this.tbGlyphAreaMin = new System.Windows.Forms.TextBox();
            this.hSBGlyphAreaMin = new System.Windows.Forms.HScrollBar();
            this.label2 = new System.Windows.Forms.Label();
            this.tbCannyHigh = new System.Windows.Forms.TextBox();
            this.hSBCannyHigh = new System.Windows.Forms.HScrollBar();
            this.label1 = new System.Windows.Forms.Label();
            this.tbCannyLow = new System.Windows.Forms.TextBox();
            this.hSBCannyLow = new System.Windows.Forms.HScrollBar();
            this.cmbBxGlyphModelNo = new System.Windows.Forms.ComboBox();
            this.label9 = new System.Windows.Forms.Label();
            this.gBoxHeadTrackingCtrl.SuspendLayout();
            this.SuspendLayout();
            // 
            // gBoxHeadTrackingCtrl
            // 
            this.gBoxHeadTrackingCtrl.Controls.Add(this.label9);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.cmbBxGlyphModelNo);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.btnSend);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.label8);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.cBoxDisplayType);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.label7);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.tbhSBConfidencePercent);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.hSBConfidencePercent);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.label5);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.tbReprojectionError);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.hSBReprojectionError);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.label6);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.tbNoIterations);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.hSBNoIterations);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.label3);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.tbGlyphAreaMax);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.hSBGlyphAreaMax);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.label4);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.tbGlyphAreaMin);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.hSBGlyphAreaMin);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.label2);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.tbCannyHigh);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.hSBCannyHigh);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.label1);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.tbCannyLow);
            this.gBoxHeadTrackingCtrl.Controls.Add(this.hSBCannyLow);
            this.gBoxHeadTrackingCtrl.Location = new System.Drawing.Point(4, 4);
            this.gBoxHeadTrackingCtrl.Name = "gBoxHeadTrackingCtrl";
            this.gBoxHeadTrackingCtrl.Size = new System.Drawing.Size(252, 470);
            this.gBoxHeadTrackingCtrl.TabIndex = 0;
            this.gBoxHeadTrackingCtrl.TabStop = false;
            this.gBoxHeadTrackingCtrl.Text = "Head Tracking Control";
            // 
            // btnSend
            // 
            this.btnSend.AutoEllipsis = true;
            this.btnSend.Location = new System.Drawing.Point(173, 430);
            this.btnSend.Name = "btnSend";
            this.btnSend.Size = new System.Drawing.Size(69, 23);
            this.btnSend.TabIndex = 23;
            this.btnSend.Text = "Send";
            this.btnSend.UseVisualStyleBackColor = true;
            this.btnSend.Click += new System.EventHandler(this.btnSend_Click);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(16, 414);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(76, 13);
            this.label8.TabIndex = 22;
            this.label8.Text = "Display Output";
            // 
            // cBoxDisplayType
            // 
            this.cBoxDisplayType.FormattingEnabled = true;
            this.cBoxDisplayType.Location = new System.Drawing.Point(13, 430);
            this.cBoxDisplayType.Name = "cBoxDisplayType";
            this.cBoxDisplayType.Size = new System.Drawing.Size(144, 21);
            this.cBoxDisplayType.TabIndex = 21;
            this.cBoxDisplayType.SelectedIndexChanged += new System.EventHandler(this.cBoxDisplayType_SelectedIndexChanged);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(13, 320);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(101, 13);
            this.label7.TabIndex = 20;
            this.label7.Text = "Confidence Percent";
            // 
            // tbhSBConfidencePercent
            // 
            this.tbhSBConfidencePercent.Location = new System.Drawing.Point(183, 339);
            this.tbhSBConfidencePercent.Name = "tbhSBConfidencePercent";
            this.tbhSBConfidencePercent.Size = new System.Drawing.Size(59, 20);
            this.tbhSBConfidencePercent.TabIndex = 19;
            // 
            // hSBConfidencePercent
            // 
            this.hSBConfidencePercent.Location = new System.Drawing.Point(13, 339);
            this.hSBConfidencePercent.Minimum = 10;
            this.hSBConfidencePercent.Name = "hSBConfidencePercent";
            this.hSBConfidencePercent.Size = new System.Drawing.Size(164, 20);
            this.hSBConfidencePercent.TabIndex = 18;
            this.hSBConfidencePercent.Value = 10;
            this.hSBConfidencePercent.Scroll += new System.Windows.Forms.ScrollEventHandler(this.hSBConfidencePercent_Scroll);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(13, 273);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(92, 13);
            this.label5.TabIndex = 17;
            this.label5.Text = "Reprojection Error";
            // 
            // tbReprojectionError
            // 
            this.tbReprojectionError.Location = new System.Drawing.Point(183, 292);
            this.tbReprojectionError.Name = "tbReprojectionError";
            this.tbReprojectionError.Size = new System.Drawing.Size(59, 20);
            this.tbReprojectionError.TabIndex = 16;
            // 
            // hSBReprojectionError
            // 
            this.hSBReprojectionError.Location = new System.Drawing.Point(13, 292);
            this.hSBReprojectionError.Minimum = 1;
            this.hSBReprojectionError.Name = "hSBReprojectionError";
            this.hSBReprojectionError.Size = new System.Drawing.Size(164, 20);
            this.hSBReprojectionError.TabIndex = 15;
            this.hSBReprojectionError.Value = 1;
            this.hSBReprojectionError.Scroll += new System.Windows.Forms.ScrollEventHandler(this.hSBReprojectionError_Scroll);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(13, 222);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(134, 13);
            this.label6.TabIndex = 14;
            this.label6.Text = "Solve Number Of Iterations";
            // 
            // tbNoIterations
            // 
            this.tbNoIterations.Location = new System.Drawing.Point(183, 241);
            this.tbNoIterations.Name = "tbNoIterations";
            this.tbNoIterations.Size = new System.Drawing.Size(59, 20);
            this.tbNoIterations.TabIndex = 13;
            // 
            // hSBNoIterations
            // 
            this.hSBNoIterations.Location = new System.Drawing.Point(13, 241);
            this.hSBNoIterations.Minimum = 1;
            this.hSBNoIterations.Name = "hSBNoIterations";
            this.hSBNoIterations.Size = new System.Drawing.Size(164, 20);
            this.hSBNoIterations.TabIndex = 12;
            this.hSBNoIterations.Value = 100;
            this.hSBNoIterations.Scroll += new System.Windows.Forms.ScrollEventHandler(this.hSBNoIterations_Scroll);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(13, 173);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(82, 13);
            this.label3.TabIndex = 11;
            this.label3.Text = "Glyph Area Max";
            // 
            // tbGlyphAreaMax
            // 
            this.tbGlyphAreaMax.Location = new System.Drawing.Point(183, 192);
            this.tbGlyphAreaMax.Name = "tbGlyphAreaMax";
            this.tbGlyphAreaMax.Size = new System.Drawing.Size(59, 20);
            this.tbGlyphAreaMax.TabIndex = 10;
            // 
            // hSBGlyphAreaMax
            // 
            this.hSBGlyphAreaMax.Location = new System.Drawing.Point(13, 192);
            this.hSBGlyphAreaMax.Maximum = 20000;
            this.hSBGlyphAreaMax.Minimum = 1000;
            this.hSBGlyphAreaMax.Name = "hSBGlyphAreaMax";
            this.hSBGlyphAreaMax.Size = new System.Drawing.Size(164, 20);
            this.hSBGlyphAreaMax.TabIndex = 9;
            this.hSBGlyphAreaMax.Value = 1000;
            this.hSBGlyphAreaMax.Scroll += new System.Windows.Forms.ScrollEventHandler(this.hSBGlyphAreaMax_Scroll);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(13, 122);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(79, 13);
            this.label4.TabIndex = 8;
            this.label4.Text = "Glyph Area Min";
            // 
            // tbGlyphAreaMin
            // 
            this.tbGlyphAreaMin.Location = new System.Drawing.Point(183, 141);
            this.tbGlyphAreaMin.Name = "tbGlyphAreaMin";
            this.tbGlyphAreaMin.Size = new System.Drawing.Size(59, 20);
            this.tbGlyphAreaMin.TabIndex = 7;
            // 
            // hSBGlyphAreaMin
            // 
            this.hSBGlyphAreaMin.Location = new System.Drawing.Point(13, 141);
            this.hSBGlyphAreaMin.Maximum = 10000;
            this.hSBGlyphAreaMin.Minimum = 500;
            this.hSBGlyphAreaMin.Name = "hSBGlyphAreaMin";
            this.hSBGlyphAreaMin.Size = new System.Drawing.Size(164, 20);
            this.hSBGlyphAreaMin.TabIndex = 6;
            this.hSBGlyphAreaMin.Value = 1000;
            this.hSBGlyphAreaMin.Scroll += new System.Windows.Forms.ScrollEventHandler(this.hSBGlyphAreaMin_Scroll);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(13, 71);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(62, 13);
            this.label2.TabIndex = 5;
            this.label2.Text = "Canny High";
            // 
            // tbCannyHigh
            // 
            this.tbCannyHigh.Location = new System.Drawing.Point(183, 90);
            this.tbCannyHigh.Name = "tbCannyHigh";
            this.tbCannyHigh.Size = new System.Drawing.Size(59, 20);
            this.tbCannyHigh.TabIndex = 4;
            // 
            // hSBCannyHigh
            // 
            this.hSBCannyHigh.Location = new System.Drawing.Point(13, 90);
            this.hSBCannyHigh.Maximum = 255;
            this.hSBCannyHigh.Minimum = 1;
            this.hSBCannyHigh.Name = "hSBCannyHigh";
            this.hSBCannyHigh.Size = new System.Drawing.Size(164, 20);
            this.hSBCannyHigh.TabIndex = 3;
            this.hSBCannyHigh.Value = 1;
            this.hSBCannyHigh.Scroll += new System.Windows.Forms.ScrollEventHandler(this.hSBCannyHigh_Scroll);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(13, 20);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(60, 13);
            this.label1.TabIndex = 2;
            this.label1.Text = "Canny Low";
            // 
            // tbCannyLow
            // 
            this.tbCannyLow.Location = new System.Drawing.Point(183, 39);
            this.tbCannyLow.Name = "tbCannyLow";
            this.tbCannyLow.Size = new System.Drawing.Size(59, 20);
            this.tbCannyLow.TabIndex = 1;
            // 
            // hSBCannyLow
            // 
            this.hSBCannyLow.Location = new System.Drawing.Point(13, 39);
            this.hSBCannyLow.Maximum = 255;
            this.hSBCannyLow.Minimum = 1;
            this.hSBCannyLow.Name = "hSBCannyLow";
            this.hSBCannyLow.Size = new System.Drawing.Size(164, 20);
            this.hSBCannyLow.TabIndex = 0;
            this.hSBCannyLow.Value = 1;
            this.hSBCannyLow.Scroll += new System.Windows.Forms.ScrollEventHandler(this.hSBCannyLow_Scroll);
            // 
            // cmbBxGlyphModelNo
            // 
            this.cmbBxGlyphModelNo.FormattingEnabled = true;
            this.cmbBxGlyphModelNo.Location = new System.Drawing.Point(183, 379);
            this.cmbBxGlyphModelNo.Name = "cmbBxGlyphModelNo";
            this.cmbBxGlyphModelNo.Size = new System.Drawing.Size(59, 21);
            this.cmbBxGlyphModelNo.TabIndex = 24;
            this.cmbBxGlyphModelNo.SelectedIndexChanged += new System.EventHandler(this.cmbBxGlyphModelNo_SelectedIndexChanged);
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(72, 382);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(105, 13);
            this.label9.TabIndex = 25;
            this.label9.Text = "Glypy Model Number";
            // 
            // HeadTrackingControlWidget
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.gBoxHeadTrackingCtrl);
            this.Name = "HeadTrackingControlWidget";
            this.Size = new System.Drawing.Size(262, 479);
            this.gBoxHeadTrackingCtrl.ResumeLayout(false);
            this.gBoxHeadTrackingCtrl.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox gBoxHeadTrackingCtrl;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox tbReprojectionError;
        private System.Windows.Forms.HScrollBar hSBReprojectionError;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox tbNoIterations;
        private System.Windows.Forms.HScrollBar hSBNoIterations;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbGlyphAreaMax;
        private System.Windows.Forms.HScrollBar hSBGlyphAreaMax;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox tbGlyphAreaMin;
        private System.Windows.Forms.HScrollBar hSBGlyphAreaMin;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbCannyHigh;
        private System.Windows.Forms.HScrollBar hSBCannyHigh;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox tbCannyLow;
        private System.Windows.Forms.HScrollBar hSBCannyLow;
        private System.Windows.Forms.Button btnSend;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.ComboBox cBoxDisplayType;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox tbhSBConfidencePercent;
        private System.Windows.Forms.HScrollBar hSBConfidencePercent;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.ComboBox cmbBxGlyphModelNo;
    }
}
