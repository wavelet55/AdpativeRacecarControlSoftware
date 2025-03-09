/* ****************************************************************
 * Athr: Harry Direen, PhD
 * Date: June 2018
 * www.DireenTech.com
 *******************************************************************/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Runtime.InteropServices;

using VidereDataLogReaderLib.DataRecords;
using VidereDataLogReaderLib.Utils;

namespace VidereDataLogReaderLib
{

    public class DataLogRecordObj
    {
        public DataRecordType_e recordType = DataRecordType_e.stdHeader;
        public DataLogRecord dataRecord;
        public StreamWriter outpFile;
        public string outpFilename;

        public DataLogRecordObj()
        {
            Clear();
        }

        public DataLogRecordObj(DataLogRecordObj dls)
        {
            recordType = dls.recordType;
            dataRecord = dls.dataRecord;
            outpFile = dls.outpFile;
        }

        void Clear()
        {
            recordType = DataRecordType_e.stdHeader;
            dataRecord = null;
            outpFile = null;
            outpFilename = null;
        }
    }

    public enum RecordStartType_e
    {
        NoRecordFound,
        HeaderRecord,
        DataRecord
    }

    public struct RecordHeaderInfo
    {
        public RecordStartType_e StartType;
        public DataRecordType_e RecordType;
        public int recordLength;

        public void Clear()
        {
            StartType = RecordStartType_e.NoRecordFound;
            RecordType = DataRecordType_e.stdHeader;
            recordLength = 0;
        }

    }

    public class DataLogReader
    {

        public string InputLogFilename = null;

        public string BaseLogFilename = null;

        public string OutpDirectoryName = null;

        public DirectoryInfo OutpDirectoryInfo = null;

        public DataLogRecord LogFileHeaderRecord;

        public ExtractOutputType_e OutpFileType = ExtractOutputType_e.json;

        private FileStream _inputLogFile = null;

        private Dictionary<DataRecordType_e, DataLogRecordObj> _outpFileRecordDict;

        private const int RecordHdrSize = 6;
        private const int MaxRecordHdrSize = 16;
        private byte[] byteArray;
        private ByteArrayReader br;

		public string DirSeparator = "/";

        public DataLogReader()
        {
            _outpFileRecordDict = new Dictionary<DataRecordType_e, DataLogRecordObj>();
            LogFileHeaderRecord = null;
            byteArray = new byte[MaxRecordHdrSize];
            br = new ByteArrayReader(byteArray, Utils.EndianOrder_e.big);

            if (IsWindowsOS())
            {
                DirSeparator = "\\";
            }
        }

        ~DataLogReader()
        {
            Close();
        }

        /// <summary>
        /// Check to see if the Operating system is Windows... if not... assume Linux.
        /// </summary>
        /// <returns></returns>
        public bool IsWindowsOS()
        {
            bool isWindows = false;
            string windir = Environment.GetEnvironmentVariable("windir");
            if (!string.IsNullOrEmpty(windir) && windir.Contains(@"\") && Directory.Exists(windir))
            {
                isWindows = true;
            }
            return isWindows;
        }

        /// <summary>
        /// Close the DataLogReader
        /// </summary>
        public void Close()
        {
            closeDataLogFile();
            if (_outpFileRecordDict != null && _outpFileRecordDict.Count > 0)
            {
                foreach( KeyValuePair<DataRecordType_e, DataLogRecordObj> kvp in _outpFileRecordDict )
                {
                    if (kvp.Value.outpFile != null)
                    {
                        kvp.Value.outpFile.Flush();
                        kvp.Value.outpFile.Close();
                    }
                }
                _outpFileRecordDict.Clear();
            }
        }

        /// <summary>
        /// Check to see if a directory exists, if it does not exist... create the directory.
        /// </summary>
        /// <param name="dirname"></param>
        /// <returns>true if there was an error creating the directory, false otherwise.</returns>
        public bool createDirectory(string dirname)
        {
            bool error = false;
            try
            {
                DirectoryInfo OutpDirectoryInfo = new DirectoryInfo(dirname);
                if (!OutpDirectoryInfo.Exists)
                {
                    OutpDirectoryInfo.Create();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Could not create directory: " + dirname
                                    + "Exception: " + ex.Message);
                error = true;
            }
            return error;
        }

        /// <summary>
        /// Open the Log File and read the header if it exists.
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public bool openDataLogFile(string filename)
        {
            bool error = true;
            try
            {
                _inputLogFile = new FileStream(filename, FileMode.Open, FileAccess.Read);
                error = false;
            }
            catch (Exception ioException)
            {
                Console.WriteLine("Cannot open sensor fusion file: " + filename
                        + "Exception: " + ioException.Message);
                return error;
            }
            if(_inputLogFile != null && _inputLogFile.Length > 0)
            {
                error = false;
            }
            return error;
        }

        public void closeDataLogFile()
        {
            if (_inputLogFile != null)
            {
                _inputLogFile.Close();
            }
        }


        private int readNextByteFromFile()
        {
            int cVal = -1;
            if (_inputLogFile != null)
            {
                try
                {
                    cVal = _inputLogFile.ReadByte();
                }
                catch (Exception ex)
                {
                    Console.Write("Exception reading log file: " + ex.Message);
                }
            }
            return cVal;
        }

        private int readBufOfDataFromFile(byte[] buf, int noBytesToRead)
        {
            int numBytesRead = 0;
            if (_inputLogFile != null)
            {
                noBytesToRead = noBytesToRead > buf.Length ? buf.Length : noBytesToRead;
                try
                {
                    numBytesRead = _inputLogFile.Read(buf, 0, noBytesToRead);
                }
                catch
                {
                    numBytesRead = -1;
                }
                if (numBytesRead < noBytesToRead)
                {
                    return 0;
                }
            }
            return numBytesRead;
        }


        private RecordHeaderInfo findRecordStart()
        {
            long fileStartPos = 0;
            int cVal = 0;
            RecordHeaderInfo rInfo = new RecordHeaderInfo();
            rInfo.Clear();
            bool foundRecordStart = false;
            bool startLetterFound = false;
            if (_inputLogFile != null )
            {
                fileStartPos = _inputLogFile.Position;
                for (int i = 0; i < 4096 && !foundRecordStart; i++)
                {
                    cVal = readNextByteFromFile();
                    if (cVal < 0)
                    {
                        //End of File Found.
                        rInfo.Clear();
                        foundRecordStart = false;
                        break;
                    }
                    if (!startLetterFound)
                    {
                        foundRecordStart = false;
                        if (cVal == 'H')
                        {
                            rInfo.StartType = RecordStartType_e.HeaderRecord;
                            startLetterFound = true;
                        }
                        else if (cVal == 'R')
                        {
                            rInfo.StartType = RecordStartType_e.DataRecord;
                            startLetterFound = true;
                        }
                    }
                    else
                    {
                        if (cVal == ':')
                        {
                            //We have found the start of a Record.
                            foundRecordStart = true;
                            break;
                        }
                        else
                        {
                            startLetterFound = false;
                        }
                    }
                }
            }
            if (foundRecordStart)
            {
                br.Reset();
                int noBytes = readBufOfDataFromFile(byteArray, RecordHdrSize);
                if (noBytes == RecordHdrSize)
                {
                    rInfo.RecordType = (DataRecordType_e)br.readUInt8();
                    rInfo.recordLength = br.readInt32();
                    cVal = br.readUInt8();
                    if (cVal != '=')
                    {
                        //Invalid record header.
                        rInfo.Clear();
                    }
                }
                else
                {
                    rInfo.Clear();
                }
            }
            return rInfo;
        }


        public bool readHeaderInfo()
        {
            bool headerFoundAndRead = false;
            RecordHeaderInfo rInfo = findRecordStart();
            if (rInfo.StartType == RecordStartType_e.HeaderRecord
                && rInfo.recordLength > 0)
            {
                LogFileHeaderRecord = DataLogRecord.createNewRecord(rInfo.RecordType);
                if (LogFileHeaderRecord != null)
                {
                    LogFileHeaderRecord.Clear();
                    int bytesRead = LogFileHeaderRecord.readRecordFromFile(_inputLogFile, rInfo.recordLength);
                    if (bytesRead > 0)
                    {
                        headerFoundAndRead = true;
                    }
                }
            }
            return headerFoundAndRead;
        }

        /// <summary>
        /// Read the next data record in the file.
        /// </summary>
        /// <returns>Returns a data record of the correct record type.
        /// Returns a null if no record or invalid record found.</returns>
        public DataLogRecord readNextDataRecord()
        {
            DataLogRecord dr = null;
            RecordHeaderInfo rInfo = findRecordStart();
            if (rInfo.StartType == RecordStartType_e.DataRecord
                && rInfo.recordLength > 0)
            {
                dr = DataLogRecord.createNewRecord(rInfo.RecordType);
                if (dr != null)
                {
                    int bytesRead = dr.readRecordFromFile(_inputLogFile, rInfo.recordLength);
                    if (bytesRead < 1)
                    {
                        //Was not able to read the record... might be end of the file.
                        dr = null;
                    }
                }
            }
            return dr;
        }

        public string createFilename(string fn, DataRecordType_e recordType)
        {
            string ofn;
            int idx = fn.LastIndexOf('.');
            if (idx > 0)
            {
                fn = fn.Substring(0, idx);
            }
            fn = fn + "_" + recordType.ToString();
            if(OutpFileType == ExtractOutputType_e.json)
                fn = fn +".json";
            else
                fn = fn +".csv";

            ofn = OutpDirectoryName + DirSeparator + fn;
            return ofn;
        }

        public DataLogRecordObj startNewOutputFile(DataLogRecord dr)
        {
            DataLogRecordObj dlo = new DataLogRecordObj();
            dlo.dataRecord = dr;
            dlo.recordType = dr.GetDataRecordType();
            dlo.outpFilename = createFilename(BaseLogFilename, dlo.recordType);
            try
            {
                dlo.outpFile = new StreamWriter(dlo.outpFilename);
                if (dlo.outpFile == null)
                {
                    Console.WriteLine("Could not open file: " + dlo.outpFilename);
                    dlo = null;
                }
            }
            catch (Exception ioException)
            {
                Console.WriteLine("Cannot open file: " + dlo.outpFilename
                        + "Exception: " + ioException.Message);
                dlo = null;
            }
            return dlo;
        }

        public bool writeDataRecordToOuptFile(DataLogRecord dr)
        {
            bool error = true;
            DataLogRecordObj dro = null;
            if (!_outpFileRecordDict.ContainsKey(dr.GetDataRecordType()))
            {
                //New Record type... we need to start a new output File
                dro = startNewOutputFile(dr);
                if (dro != null)
                {
                    _outpFileRecordDict[dro.recordType] = dro;
                    //Add Header Info to File
                    if (LogFileHeaderRecord != null)
                    {
                        LogFileHeaderRecord.writeColumnNames(dro.outpFile, OutpFileType);
                        LogFileHeaderRecord.writeRecordToFile(dro.outpFile, OutpFileType);
                        dro.dataRecord.writeColumnNames(dro.outpFile, OutpFileType);
                    }
                }
                else
                {
                    return true;
                }
            }
            dro = _outpFileRecordDict[dr.GetDataRecordType()];
            dro.dataRecord = dr;
            if (dr.writeRecordToFile(dro.outpFile, OutpFileType) > 0)
            {
                error = false;
            }
            return error;
        }

        /// <summary>
        /// Extract a Log File.
        /// A log file is a binary formated file that can contain a range
        /// off different data record types.  A new output file will be created
        /// for each record type.  This will make using the output log file
        /// easier to parse and use.  Each output file will contain the name of the
        /// original file, the record type, and be teminated in the ouput type: csv or json.
        /// </summary>
        /// <param name="logFileName">Full pathname of the log file.</param>
        /// <param name="outputDirectory">directory to store the extracted ouput files into.</param>
        /// <param name="outpType"></param>
        /// <returns>Number of extracted files.  A negative number indicates there was an error
        /// extracting one or more of the files.</returns>
        public int extractLogFile(string logFileName, string outputDirectory, ExtractOutputType_e outpType = ExtractOutputType_e.json)
        {
            int numberOfFiles = 0;
            bool endOfRecords = false;
            InputLogFilename = logFileName;
            OutpFileType = outpType;
            OutpDirectoryName = outputDirectory;
            _outpFileRecordDict.Clear();

            int idx = logFileName.LastIndexOf(DirSeparator);
            if (idx > 0 && logFileName.Length > idx)
                BaseLogFilename = logFileName.Substring(idx + 1);
            else
                BaseLogFilename = logFileName;

            if(outputDirectory != null && outputDirectory != "")
            {
                if(createDirectory(outputDirectory))
                    return 0;
            }

            if(openDataLogFile(logFileName))
            {
                return 0;
            }

            if(readHeaderInfo())
            {
                while (!endOfRecords)
                {
                    DataLogRecord dr = readNextDataRecord();
                    if (dr != null)
                    {
                        if (writeDataRecordToOuptFile(dr))
                        {
                            //Error writting out records... quit.
                            endOfRecords = true;
                        }
                    }
                    else
                    {
                        endOfRecords = true;
                    }
                }
                numberOfFiles = _outpFileRecordDict.Count;
            }
            Close();
            return numberOfFiles;
        }
    }
}
