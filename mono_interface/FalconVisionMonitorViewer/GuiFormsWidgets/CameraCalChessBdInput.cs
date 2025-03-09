using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace FalconVisionMonitorViewer.GuiFormsWidgets
{
    public partial class CameraCalChessBdInput : UserControl
    {
        public int NumberOfRows = 7;
        public int NumberOfCols = 6;
        public double SquareSizeMilliMeters = 25.4;
        public bool ParameterError = false;


        public CameraCalChessBdInput()
        {
            InitializeComponent();
            cbMeasUnits.Items.Add("inches");
            cbMeasUnits.Items.Add("mm");
            cbMeasUnits.Items.Add("cm");
            cbMeasUnits.SelectedIndex = 0;
        }

        public bool ReadParameters()
        {
            ParameterError = false;
            double sqsize = 1.0;
            if (int.TryParse(tbCBNoCols.Text, out NumberOfCols))
            {
                NumberOfCols = NumberOfCols < 1 ? 1 : NumberOfCols > 100 ? 100 : NumberOfCols;
            }
            else
            {
                ParameterError = true;
                NumberOfCols = 6;
            }
            tbCBNoCols.Text = NumberOfCols.ToString();

            if (int.TryParse(tbCBNoRows.Text, out NumberOfRows))
            {
                NumberOfRows = NumberOfRows < 1 ? 1 : NumberOfRows > 100 ? 100 : NumberOfRows;
            }
            else
            {
                ParameterError = true;
                NumberOfRows = 7;
            }
            tbCBNoRows.Text = NumberOfRows.ToString();

            int unit = cbMeasUnits.SelectedIndex;
            if (double.TryParse(tbSqSize.Text, out sqsize))
            {
                SquareSizeMilliMeters = sqsize < 1e-3 ? 1e-3 : sqsize > 1000.0 ? 1000.0 : sqsize;
                if (unit == 0)  //Inches
                    SquareSizeMilliMeters = sqsize * 25.4;  //inches to mm;
                else if (unit == 2)
                    SquareSizeMilliMeters = sqsize * 10.0;  //cm to mm;
            }
            else
            {
                ParameterError = true;
                sqsize = 1.0;
                SquareSizeMilliMeters = 25.4;
            }
            tbSqSize.Text = sqsize.ToString("0.000000");
            
            return ParameterError;
        }



    }
}
