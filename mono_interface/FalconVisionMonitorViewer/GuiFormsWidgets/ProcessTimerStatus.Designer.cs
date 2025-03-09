namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class ProcessTimerStatus
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
            this.gbTimerStatus = new System.Windows.Forms.GroupBox();
            this.tbTimer1 = new System.Windows.Forms.TextBox();
            this.tbTimer2 = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.cbTimer1Units = new System.Windows.Forms.ComboBox();
            this.cbTimer2Units = new System.Windows.Forms.ComboBox();
            this.gbTimerStatus.SuspendLayout();
            this.SuspendLayout();
            // 
            // gbTimerStatus
            // 
            this.gbTimerStatus.Controls.Add(this.cbTimer2Units);
            this.gbTimerStatus.Controls.Add(this.cbTimer1Units);
            this.gbTimerStatus.Controls.Add(this.label2);
            this.gbTimerStatus.Controls.Add(this.label1);
            this.gbTimerStatus.Controls.Add(this.tbTimer2);
            this.gbTimerStatus.Controls.Add(this.tbTimer1);
            this.gbTimerStatus.Location = new System.Drawing.Point(4, 4);
            this.gbTimerStatus.Name = "gbTimerStatus";
            this.gbTimerStatus.Size = new System.Drawing.Size(184, 77);
            this.gbTimerStatus.TabIndex = 0;
            this.gbTimerStatus.TabStop = false;
            this.gbTimerStatus.Text = "Timer Status";
            // 
            // tbTimer1
            // 
            this.tbTimer1.Location = new System.Drawing.Point(30, 19);
            this.tbTimer1.Name = "tbTimer1";
            this.tbTimer1.Size = new System.Drawing.Size(71, 20);
            this.tbTimer1.TabIndex = 0;
            this.tbTimer1.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // tbTimer2
            // 
            this.tbTimer2.Location = new System.Drawing.Point(30, 45);
            this.tbTimer2.Name = "tbTimer2";
            this.tbTimer2.Size = new System.Drawing.Size(71, 20);
            this.tbTimer2.TabIndex = 1;
            this.tbTimer2.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(4, 22);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(20, 13);
            this.label1.TabIndex = 2;
            this.label1.Text = "T1";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(4, 48);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(20, 13);
            this.label2.TabIndex = 3;
            this.label2.Text = "T2";
            // 
            // cbTimer1Units
            // 
            this.cbTimer1Units.FormattingEnabled = true;
            this.cbTimer1Units.Location = new System.Drawing.Point(116, 19);
            this.cbTimer1Units.Name = "cbTimer1Units";
            this.cbTimer1Units.Size = new System.Drawing.Size(54, 21);
            this.cbTimer1Units.TabIndex = 4;
            // 
            // cbTimer2Units
            // 
            this.cbTimer2Units.FormattingEnabled = true;
            this.cbTimer2Units.Location = new System.Drawing.Point(116, 45);
            this.cbTimer2Units.Name = "cbTimer2Units";
            this.cbTimer2Units.Size = new System.Drawing.Size(54, 21);
            this.cbTimer2Units.TabIndex = 5;
            // 
            // ProcessTimerStatus
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.gbTimerStatus);
            this.Name = "ProcessTimerStatus";
            this.Size = new System.Drawing.Size(197, 86);
            this.gbTimerStatus.ResumeLayout(false);
            this.gbTimerStatus.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox gbTimerStatus;
        private System.Windows.Forms.ComboBox cbTimer2Units;
        private System.Windows.Forms.ComboBox cbTimer1Units;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox tbTimer2;
        private System.Windows.Forms.TextBox tbTimer1;
    }
}
