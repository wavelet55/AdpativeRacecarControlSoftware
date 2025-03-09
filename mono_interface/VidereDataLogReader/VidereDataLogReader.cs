/* ****************************************************************
 * Athr: Harry Direen, PhD
 * Date: June 2018
 * www.DireenTech.com
 *******************************************************************/

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using VidereDataLogReaderLib;

namespace VidereDataLogReader
{
    public partial class VidereDataLogReader : Form
    {
        private DataLogReader _dlogReader;

        private List<FileInfo> _dataLogFiles = null;

        private bool _ignoreChange = false;

        public VidereDataLogReader()
        {
            InitializeComponent();
            _dlogReader = new DataLogReader();

            _ignoreChange = true;
            cbLogFileType.Items.Add("Data Logs");
            cbLogFileType.Items.Add("Image Plus Metadata");
            cbLogFileType.SelectedIndex = 0;
            _ignoreChange = false;

            setOutputTypeComboBox(0);

            tbLogFileExt.Text = "dat";
        }

        private void setOutputTypeComboBox(int logType)
        {
            cbxOutputType.Items.Clear();
            if (logType == 1)
            {
                cbxOutputType.Items.Add("avi");
            }
            else
            {
                cbxOutputType.Items.Add("cvs");
            }
            _ignoreChange = true;
            cbxOutputType.SelectedIndex = 0;
            _ignoreChange = false;
        }

        private List<FileInfo> filterFilesByExtention(FileInfo[] files, string fileExt)
        {
            bool allFiles = false;
            if(fileExt == null || fileExt.Length < 1 || fileExt.Contains('*') )
            {
                allFiles = true;
            }
            else
            {
                if(!fileExt.StartsWith("."))
                {
                    fileExt = "." + fileExt;
                }
            }

            List<FileInfo> fileList = new List<FileInfo>();
            if (files != null && files.Length > 0)
            {
                foreach (FileInfo file in files)
                {
                    if (allFiles || file.Name.EndsWith(fileExt))
                    {
                        fileList.Add(file);
                    }
                }
            }
            return fileList;
        }

        private List<FileInfo> getDataLogFileList(string directory)
        {
            List<FileInfo> logFiles = null;
            DirectoryInfo dinfo = new DirectoryInfo(directory);
            if (!dinfo.Exists)
            {
                MessageBox.Show("Directory: " + directory
                    + " does not exist.");
                return logFiles;
            }

            FileInfo[] files = dinfo.GetFiles();
            if (files != null && files.Length > 0)
            {
                logFiles = filterFilesByExtention(files, tbLogFileExt.Text);
                if (logFiles != null)
                {
                    tbNumberOfLogFiles.Text = logFiles.Count.ToString();
                }
            }
            return logFiles;
        }

        private void btnBrowseLogFileDir_Click(object sender, EventArgs e)
        {
            string initFolder = @"";
            _dataLogFiles = null;
            if (tbLogFilesDirectory.Text != null && tbLogFilesDirectory.Text != "")
            {
                initFolder = tbLogFilesDirectory.Text;
            }
            FolderBrowserDialog fdlg = new FolderBrowserDialog();
            fdlg.SelectedPath = initFolder;
            fdlg.ShowNewFolderButton = false;
            if (fdlg.ShowDialog() == DialogResult.OK)
            {
                tbLogFilesDirectory.Text = fdlg.SelectedPath;
                if (tbOutputDirectory.Text != null && tbOutputDirectory.Text.Length < 1)
                {
                    tbOutputDirectory.Text = tbLogFilesDirectory.Text;
                }
                _dataLogFiles = getDataLogFileList(tbLogFilesDirectory.Text);
            }
        }

        private void btnBrowseOutputDir_Click(object sender, EventArgs e)
        {
            string initFolder = @"";
            if (tbOutputDirectory.Text != null && tbOutputDirectory.Text != "")
            {
                initFolder = tbOutputDirectory.Text;
            }
            FolderBrowserDialog fdlg = new FolderBrowserDialog();
            fdlg.SelectedPath = initFolder;
            fdlg.ShowNewFolderButton = false;
            if (fdlg.ShowDialog() == DialogResult.OK)
            {
                tbOutputDirectory.Text = fdlg.SelectedPath;
            }
        }

        private void btnExtractLogFiles_Click(object sender, EventArgs e)
        {
            int numOutputFiles = 0;
            if (tbLogFilesDirectory.Text != null && tbLogFilesDirectory.Text.Length > 0)
            {
                _dataLogFiles = getDataLogFileList(tbLogFilesDirectory.Text);
                if (_dataLogFiles != null && _dataLogFiles.Count > 0
                    && tbOutputDirectory.Text != null && tbOutputDirectory.Text.Length > 0)
                {
                    if (cbxOutputType.SelectedIndex == 0)
                    {
                        foreach (FileInfo logFile in _dataLogFiles)
                        {
                            numOutputFiles += _dlogReader.extractLogFile(logFile.FullName, tbOutputDirectory.Text,
                                VidereDataLogReaderLib.DataRecords.ExtractOutputType_e.csv);
                        }
                    }
                    else if (cbxOutputType.SelectedIndex == 1)
                    {


                    }
                }
            }
            tbNumberOutputFiles.Text = numOutputFiles.ToString();
        }

        private void cbxOutputType_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void cbLogFileType_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (!_ignoreChange)
            {
                setOutputTypeComboBox(cbLogFileType.SelectedIndex);
                if (cbLogFileType.SelectedIndex == 1)
                {
                    tbLogFileExt.Text = "imd";
                }
                else
                {
                    tbLogFileExt.Text = "dat";
                }
            }
        }
    }
}
