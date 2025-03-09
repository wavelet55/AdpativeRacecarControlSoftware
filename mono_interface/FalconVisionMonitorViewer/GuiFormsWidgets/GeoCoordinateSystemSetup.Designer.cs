namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    partial class GeoCoordinateSystemSetup
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
            this.grpbxGeoCoordSetup = new System.Windows.Forms.GroupBox();
            this.cbxGeoCConvType = new System.Windows.Forms.ComboBox();
            this.btn_GetGeoCoordSetup = new System.Windows.Forms.Button();
            this.btnXYToLatLonConv = new System.Windows.Forms.Button();
            this.btnLatLonToXYConv = new System.Windows.Forms.Button();
            this.tb_GCSC_Y = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.tb_GCSC_X = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.tb_GCSC_Lon = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.tb_GCSC_Lat = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.btnSetGeoCoords = new System.Windows.Forms.Button();
            this.label4 = new System.Windows.Forms.Label();
            this.tbGCS_GroundAltMSL = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.tbGCS_CenterLon = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.tbGCS_CenterLat = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.grpbxGeoCoordSetup.SuspendLayout();
            this.SuspendLayout();
            // 
            // grpbxGeoCoordSetup
            // 
            this.grpbxGeoCoordSetup.Controls.Add(this.cbxGeoCConvType);
            this.grpbxGeoCoordSetup.Controls.Add(this.btn_GetGeoCoordSetup);
            this.grpbxGeoCoordSetup.Controls.Add(this.btnXYToLatLonConv);
            this.grpbxGeoCoordSetup.Controls.Add(this.btnLatLonToXYConv);
            this.grpbxGeoCoordSetup.Controls.Add(this.tb_GCSC_Y);
            this.grpbxGeoCoordSetup.Controls.Add(this.label7);
            this.grpbxGeoCoordSetup.Controls.Add(this.tb_GCSC_X);
            this.grpbxGeoCoordSetup.Controls.Add(this.label8);
            this.grpbxGeoCoordSetup.Controls.Add(this.tb_GCSC_Lon);
            this.grpbxGeoCoordSetup.Controls.Add(this.label5);
            this.grpbxGeoCoordSetup.Controls.Add(this.tb_GCSC_Lat);
            this.grpbxGeoCoordSetup.Controls.Add(this.label6);
            this.grpbxGeoCoordSetup.Controls.Add(this.btnSetGeoCoords);
            this.grpbxGeoCoordSetup.Controls.Add(this.label4);
            this.grpbxGeoCoordSetup.Controls.Add(this.tbGCS_GroundAltMSL);
            this.grpbxGeoCoordSetup.Controls.Add(this.label3);
            this.grpbxGeoCoordSetup.Controls.Add(this.tbGCS_CenterLon);
            this.grpbxGeoCoordSetup.Controls.Add(this.label2);
            this.grpbxGeoCoordSetup.Controls.Add(this.tbGCS_CenterLat);
            this.grpbxGeoCoordSetup.Controls.Add(this.label1);
            this.grpbxGeoCoordSetup.Location = new System.Drawing.Point(3, 3);
            this.grpbxGeoCoordSetup.Name = "grpbxGeoCoordSetup";
            this.grpbxGeoCoordSetup.Size = new System.Drawing.Size(385, 225);
            this.grpbxGeoCoordSetup.TabIndex = 0;
            this.grpbxGeoCoordSetup.TabStop = false;
            this.grpbxGeoCoordSetup.Text = "Geo-Coordinate System Setup";
            this.grpbxGeoCoordSetup.Enter += new System.EventHandler(this.grpbxGeoCoordSetup_Enter);
            // 
            // cbxGeoCConvType
            // 
            this.cbxGeoCConvType.ForeColor = System.Drawing.SystemColors.WindowText;
            this.cbxGeoCConvType.FormattingEnabled = true;
            this.cbxGeoCConvType.IntegralHeight = false;
            this.cbxGeoCConvType.Location = new System.Drawing.Point(16, 92);
            this.cbxGeoCConvType.Name = "cbxGeoCConvType";
            this.cbxGeoCConvType.RightToLeft = System.Windows.Forms.RightToLeft.Yes;
            this.cbxGeoCConvType.Size = new System.Drawing.Size(111, 21);
            this.cbxGeoCConvType.TabIndex = 25;
            // 
            // btn_GetGeoCoordSetup
            // 
            this.btn_GetGeoCoordSetup.Location = new System.Drawing.Point(267, 90);
            this.btn_GetGeoCoordSetup.Name = "btn_GetGeoCoordSetup";
            this.btn_GetGeoCoordSetup.Size = new System.Drawing.Size(99, 23);
            this.btn_GetGeoCoordSetup.TabIndex = 24;
            this.btn_GetGeoCoordSetup.Text = "Get Coordinates";
            this.btn_GetGeoCoordSetup.UseVisualStyleBackColor = true;
            this.btn_GetGeoCoordSetup.Click += new System.EventHandler(this.btn_GetGeoCoordSetup_Click);
            // 
            // btnXYToLatLonConv
            // 
            this.btnXYToLatLonConv.Location = new System.Drawing.Point(52, 186);
            this.btnXYToLatLonConv.Name = "btnXYToLatLonConv";
            this.btnXYToLatLonConv.Size = new System.Drawing.Size(99, 23);
            this.btnXYToLatLonConv.TabIndex = 23;
            this.btnXYToLatLonConv.Text = "X-Y To Lat/Lon";
            this.btnXYToLatLonConv.UseVisualStyleBackColor = true;
            this.btnXYToLatLonConv.Click += new System.EventHandler(this.btnXYToLatLonConv_Click);
            // 
            // btnLatLonToXYConv
            // 
            this.btnLatLonToXYConv.Location = new System.Drawing.Point(52, 145);
            this.btnLatLonToXYConv.Name = "btnLatLonToXYConv";
            this.btnLatLonToXYConv.Size = new System.Drawing.Size(99, 23);
            this.btnLatLonToXYConv.TabIndex = 22;
            this.btnLatLonToXYConv.Text = "Lat/Lon To X-Y";
            this.btnLatLonToXYConv.UseMnemonic = false;
            this.btnLatLonToXYConv.UseVisualStyleBackColor = true;
            this.btnLatLonToXYConv.Click += new System.EventHandler(this.btnLatLonToXYConv_Click);
            // 
            // tb_GCSC_Y
            // 
            this.tb_GCSC_Y.Location = new System.Drawing.Point(266, 186);
            this.tb_GCSC_Y.Name = "tb_GCSC_Y";
            this.tb_GCSC_Y.Size = new System.Drawing.Size(100, 20);
            this.tb_GCSC_Y.TabIndex = 21;
            this.tb_GCSC_Y.Text = "0";
            this.tb_GCSC_Y.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(287, 169);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(54, 13);
            this.label7.TabIndex = 20;
            this.label7.Text = "Y (meters)";
            // 
            // tb_GCSC_X
            // 
            this.tb_GCSC_X.Location = new System.Drawing.Point(157, 186);
            this.tb_GCSC_X.Name = "tb_GCSC_X";
            this.tb_GCSC_X.Size = new System.Drawing.Size(99, 20);
            this.tb_GCSC_X.TabIndex = 19;
            this.tb_GCSC_X.Text = "0";
            this.tb_GCSC_X.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(180, 169);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(54, 13);
            this.label8.TabIndex = 18;
            this.label8.Text = "X (meters)";
            // 
            // tb_GCSC_Lon
            // 
            this.tb_GCSC_Lon.Location = new System.Drawing.Point(266, 145);
            this.tb_GCSC_Lon.Name = "tb_GCSC_Lon";
            this.tb_GCSC_Lon.Size = new System.Drawing.Size(100, 20);
            this.tb_GCSC_Lon.TabIndex = 17;
            this.tb_GCSC_Lon.Text = "0";
            this.tb_GCSC_Lon.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(263, 128);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(103, 13);
            this.label5.TabIndex = 16;
            this.label5.Text = "Longitude (Degrees)";
            // 
            // tb_GCSC_Lat
            // 
            this.tb_GCSC_Lat.Location = new System.Drawing.Point(157, 145);
            this.tb_GCSC_Lat.Name = "tb_GCSC_Lat";
            this.tb_GCSC_Lat.Size = new System.Drawing.Size(99, 20);
            this.tb_GCSC_Lat.TabIndex = 15;
            this.tb_GCSC_Lat.Text = "0";
            this.tb_GCSC_Lat.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(154, 128);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(94, 13);
            this.label6.TabIndex = 14;
            this.label6.Text = "Latitude (Degrees)";
            // 
            // btnSetGeoCoords
            // 
            this.btnSetGeoCoords.Location = new System.Drawing.Point(157, 90);
            this.btnSetGeoCoords.Name = "btnSetGeoCoords";
            this.btnSetGeoCoords.Size = new System.Drawing.Size(99, 23);
            this.btnSetGeoCoords.TabIndex = 13;
            this.btnSetGeoCoords.Text = "Set Coordinates";
            this.btnSetGeoCoords.UseVisualStyleBackColor = true;
            this.btnSetGeoCoords.Click += new System.EventHandler(this.btnSetGeoCoords_Click);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label4.Location = new System.Drawing.Point(79, 24);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(220, 13);
            this.label4.TabIndex = 12;
            this.label4.Text = "Center Of Mission Operational Region";
            this.label4.TextAlign = System.Drawing.ContentAlignment.BottomRight;
            // 
            // tbGCS_GroundAltMSL
            // 
            this.tbGCS_GroundAltMSL.Location = new System.Drawing.Point(258, 54);
            this.tbGCS_GroundAltMSL.Name = "tbGCS_GroundAltMSL";
            this.tbGCS_GroundAltMSL.Size = new System.Drawing.Size(108, 20);
            this.tbGCS_GroundAltMSL.TabIndex = 11;
            this.tbGCS_GroundAltMSL.Text = "2200.0";
            this.tbGCS_GroundAltMSL.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(255, 37);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(111, 13);
            this.label3.TabIndex = 10;
            this.label3.Text = "Ground Altitude (MSL)";
            // 
            // tbGCS_CenterLon
            // 
            this.tbGCS_CenterLon.Location = new System.Drawing.Point(121, 54);
            this.tbGCS_CenterLon.Name = "tbGCS_CenterLon";
            this.tbGCS_CenterLon.Size = new System.Drawing.Size(100, 20);
            this.tbGCS_CenterLon.TabIndex = 9;
            this.tbGCS_CenterLon.Text = "-104.85";
            this.tbGCS_CenterLon.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(118, 37);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(103, 13);
            this.label2.TabIndex = 8;
            this.label2.Text = "Longitude (Degrees)";
            // 
            // tbGCS_CenterLat
            // 
            this.tbGCS_CenterLat.Location = new System.Drawing.Point(16, 54);
            this.tbGCS_CenterLat.Name = "tbGCS_CenterLat";
            this.tbGCS_CenterLat.Size = new System.Drawing.Size(99, 20);
            this.tbGCS_CenterLat.TabIndex = 7;
            this.tbGCS_CenterLat.Text = "39.035";
            this.tbGCS_CenterLat.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(13, 37);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(94, 13);
            this.label1.TabIndex = 6;
            this.label1.Text = "Latitude (Degrees)";
            // 
            // GeoCoordinateSystemSetup
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.grpbxGeoCoordSetup);
            this.Name = "GeoCoordinateSystemSetup";
            this.Size = new System.Drawing.Size(393, 233);
            this.grpbxGeoCoordSetup.ResumeLayout(false);
            this.grpbxGeoCoordSetup.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox grpbxGeoCoordSetup;
        private System.Windows.Forms.TextBox tbGCS_GroundAltMSL;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox tbGCS_CenterLon;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tbGCS_CenterLat;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button btnXYToLatLonConv;
        private System.Windows.Forms.Button btnLatLonToXYConv;
        private System.Windows.Forms.TextBox tb_GCSC_Y;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox tb_GCSC_X;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox tb_GCSC_Lon;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox tb_GCSC_Lat;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Button btnSetGeoCoords;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.ComboBox cbxGeoCConvType;
        private System.Windows.Forms.Button btn_GetGeoCoordSetup;

    }
}
