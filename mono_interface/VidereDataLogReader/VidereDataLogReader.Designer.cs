namespace VidereDataLogReader
{
    partial class VidereDataLogReader
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

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.tbLogFilesDirectory = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.btnBrowseLogFileDir = new System.Windows.Forms.Button();
            this.tbLogFileExt = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.tbNumberOfLogFiles = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.btnBrowseOutputDir = new System.Windows.Forms.Button();
            this.label5 = new System.Windows.Forms.Label();
            this.tbOutputDirectory = new System.Windows.Forms.TextBox();
            this.cbxOutputType = new System.Windows.Forms.ComboBox();
            this.btnExtractLogFiles = new System.Windows.Forms.Button();
            this.label6 = new System.Windows.Forms.Label();
            this.tbNumberOutputFiles = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.tbMessages = new System.Windows.Forms.TextBox();
            this.cbLogFileType = new System.Windows.Forms.ComboBox();
            this.label8 = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // tbLogFilesDirectory
            // 
            this.tbLogFilesDirectory.Location = new System.Drawing.Point(31, 85);
            this.tbLogFilesDirectory.Name = "tbLogFilesDirectory";
            this.tbLogFilesDirectory.Size = new System.Drawing.Size(312, 20);
            this.tbLogFilesDirectory.TabIndex = 0;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(31, 66);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(100, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Log File(s) Directory";
            // 
            // btnBrowseLogFileDir
            // 
            this.btnBrowseLogFileDir.Location = new System.Drawing.Point(402, 83);
            this.btnBrowseLogFileDir.Name = "btnBrowseLogFileDir";
            this.btnBrowseLogFileDir.Size = new System.Drawing.Size(75, 23);
            this.btnBrowseLogFileDir.TabIndex = 2;
            this.btnBrowseLogFileDir.Text = "Browse...";
            this.btnBrowseLogFileDir.UseVisualStyleBackColor = true;
            this.btnBrowseLogFileDir.Click += new System.EventHandler(this.btnBrowseLogFileDir_Click);
            // 
            // tbLogFileExt
            // 
            this.tbLogFileExt.Location = new System.Drawing.Point(349, 85);
            this.tbLogFileExt.Name = "tbLogFileExt";
            this.tbLogFileExt.Size = new System.Drawing.Size(47, 20);
            this.tbLogFileExt.TabIndex = 3;
            this.tbLogFileExt.Text = "dat";
            this.tbLogFileExt.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(346, 66);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(22, 13);
            this.label2.TabIndex = 4;
            this.label2.Text = "Ext";
            // 
            // tbNumberOfLogFiles
            // 
            this.tbNumberOfLogFiles.Location = new System.Drawing.Point(316, 112);
            this.tbNumberOfLogFiles.Name = "tbNumberOfLogFiles";
            this.tbNumberOfLogFiles.ReadOnly = true;
            this.tbNumberOfLogFiles.Size = new System.Drawing.Size(80, 20);
            this.tbNumberOfLogFiles.TabIndex = 5;
            this.tbNumberOfLogFiles.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(210, 115);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(104, 13);
            this.label3.TabIndex = 6;
            this.label3.Text = "Number of Log Files:";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(31, 219);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(85, 13);
            this.label4.TabIndex = 11;
            this.label4.Text = "Output File Type";
            // 
            // btnBrowseOutputDir
            // 
            this.btnBrowseOutputDir.Location = new System.Drawing.Point(358, 171);
            this.btnBrowseOutputDir.Name = "btnBrowseOutputDir";
            this.btnBrowseOutputDir.Size = new System.Drawing.Size(75, 23);
            this.btnBrowseOutputDir.TabIndex = 9;
            this.btnBrowseOutputDir.Text = "Browse...";
            this.btnBrowseOutputDir.UseVisualStyleBackColor = true;
            this.btnBrowseOutputDir.Click += new System.EventHandler(this.btnBrowseOutputDir_Click);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(31, 154);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(84, 13);
            this.label5.TabIndex = 8;
            this.label5.Text = "Output Directory";
            // 
            // tbOutputDirectory
            // 
            this.tbOutputDirectory.Location = new System.Drawing.Point(31, 173);
            this.tbOutputDirectory.Name = "tbOutputDirectory";
            this.tbOutputDirectory.Size = new System.Drawing.Size(312, 20);
            this.tbOutputDirectory.TabIndex = 7;
            // 
            // cbxOutputType
            // 
            this.cbxOutputType.FormattingEnabled = true;
            this.cbxOutputType.Location = new System.Drawing.Point(34, 236);
            this.cbxOutputType.Name = "cbxOutputType";
            this.cbxOutputType.Size = new System.Drawing.Size(97, 21);
            this.cbxOutputType.TabIndex = 12;
            this.cbxOutputType.SelectedIndexChanged += new System.EventHandler(this.cbxOutputType_SelectedIndexChanged);
            // 
            // btnExtractLogFiles
            // 
            this.btnExtractLogFiles.Location = new System.Drawing.Point(153, 236);
            this.btnExtractLogFiles.Name = "btnExtractLogFiles";
            this.btnExtractLogFiles.Size = new System.Drawing.Size(75, 23);
            this.btnExtractLogFiles.TabIndex = 13;
            this.btnExtractLogFiles.Text = "Extract";
            this.btnExtractLogFiles.UseVisualStyleBackColor = true;
            this.btnExtractLogFiles.Click += new System.EventHandler(this.btnExtractLogFiles_Click);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(26, 286);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(118, 13);
            this.label6.TabIndex = 15;
            this.label6.Text = "Number of Output Files:";
            // 
            // tbNumberOutputFiles
            // 
            this.tbNumberOutputFiles.Location = new System.Drawing.Point(148, 283);
            this.tbNumberOutputFiles.Name = "tbNumberOutputFiles";
            this.tbNumberOutputFiles.ReadOnly = true;
            this.tbNumberOutputFiles.Size = new System.Drawing.Size(80, 20);
            this.tbNumberOutputFiles.TabIndex = 14;
            this.tbNumberOutputFiles.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(26, 321);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(58, 13);
            this.label7.TabIndex = 17;
            this.label7.Text = "Messages:";
            // 
            // tbMessages
            // 
            this.tbMessages.Location = new System.Drawing.Point(90, 318);
            this.tbMessages.Name = "tbMessages";
            this.tbMessages.ReadOnly = true;
            this.tbMessages.Size = new System.Drawing.Size(253, 20);
            this.tbMessages.TabIndex = 16;
            this.tbMessages.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // cbLogFileType
            // 
            this.cbLogFileType.FormattingEnabled = true;
            this.cbLogFileType.Location = new System.Drawing.Point(29, 28);
            this.cbLogFileType.Name = "cbLogFileType";
            this.cbLogFileType.Size = new System.Drawing.Size(249, 21);
            this.cbLogFileType.TabIndex = 18;
            this.cbLogFileType.SelectedIndexChanged += new System.EventHandler(this.cbLogFileType_SelectedIndexChanged);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(31, 9);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(71, 13);
            this.label8.TabIndex = 19;
            this.label8.Text = "Log File Type";
            // 
            // VidereDataLogReader
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(548, 370);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.cbLogFileType);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.tbMessages);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.tbNumberOutputFiles);
            this.Controls.Add(this.btnExtractLogFiles);
            this.Controls.Add(this.cbxOutputType);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.btnBrowseOutputDir);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.tbOutputDirectory);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.tbNumberOfLogFiles);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.tbLogFileExt);
            this.Controls.Add(this.btnBrowseLogFileDir);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.tbLogFilesDirectory);
            this.Name = "VidereDataLogReader";
            this.Text = "Videre Data Log Reader";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox tbLogFilesDirectory;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button btnBrowseLogFileDir;
        private System.Windows.Forms.TextBox tbLogFileExt;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbNumberOfLogFiles;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Button btnBrowseOutputDir;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox tbOutputDirectory;
        private System.Windows.Forms.ComboBox cbxOutputType;
        private System.Windows.Forms.Button btnExtractLogFiles;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox tbNumberOutputFiles;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox tbMessages;
        private System.Windows.Forms.ComboBox cbLogFileType;
        private System.Windows.Forms.Label label8;
    }
}

