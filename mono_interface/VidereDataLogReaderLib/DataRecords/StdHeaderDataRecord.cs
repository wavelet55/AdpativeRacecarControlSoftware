using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using VidereDataLogReaderLib.Utils;

namespace VidereDataLogReaderLib.DataRecords
{
    public class StdHeaderDataRecord : DataLogRecord
    {
        public string DataLogFileName;

        public UInt32 VersionNumber;

        //The Endianess of the Computer generating the Image Plus Metadata
        //File.  This information is required because the Metadata information
        //is stored directly as a blob, and the Endianess of the data will
        //depend on the computer operating system.
        // EndianOrder_e ComputerEndianess;

        public DateTime LogDateTime;

        //TimeStamp in seconds... all records
        //have a timestamp associated with the recorded data.
        //Within the system the timestamps must be synchronous so that
        //recorded data can be ordered by time.
        public double TimeStampSec;

        //Private members that are not serialized.
        private const int MaxLogRecordSize = 128;
        private byte[] byteArray;
        private ByteArrayReader br;

        public StdHeaderDataRecord()
        {
            byteArray = new byte[MaxLogRecordSize];
            br = new ByteArrayReader(byteArray, Utils.EndianOrder_e.big);
            Clear();
        }

        override public DataRecordType_e GetDataRecordType()
        {
            return DataRecordType_e.stdHeader;
        }


        override public void Clear()
        {
            DataLogFileName = "";
            VersionNumber = 0;
            //ComputerEndianess = EndianOrder_e.BigEndian;
            TimeStampSec = 0;
            LogDateTime = DateTime.MinValue;
        }

        /// <summary>
        /// Read and deserialize record from the file.
        /// The record type must already be established as the 
        /// concrete object must be of the record type in the file
        /// </summary>
        /// <param name="file">The filestreem that is already at the location of the record.</param>
        /// <param name="recordSize"></param>
        /// <returns></returns>
        override public int readRecordFromFile(FileStream file, int recordSize)
        {
            int noBytes = 0;
            br.Reset();
            noBytes = readBufOfDataFromFile(file, byteArray, recordSize);
            if (noBytes == recordSize)
            {
                int size = br.readInt16();  //string size
                DataLogFileName = br.readString(size);
                VersionNumber = br.readUInt32();
                EndianOrder_e computerEndianess = (EndianOrder_e)br.readUInt8();
                TimeStampSec = br.readDouble();

                int Year = (int)br.readInt16();
                int Month = (int)br.readUInt8() + 1;
                int Day = (int)br.readUInt8();
                int Hour = (int)br.readUInt8();
                int Minute = (int)br.readUInt8();
                int second = (int)br.readUInt8();
                LogDateTime = new DateTime(Year, Month, Day, Hour, Minute, second);
                noBytes = recordSize;
            }
            else
            {
                noBytes = 0;
            }
            return noBytes;
        }

        /// <summary>
        /// Write this record out to the output file in the given format.
        /// </summary>
        /// <param name="file">File steam set to the location of the next record to be written.</param>
        /// <param name="outputType"></param>
        /// <returns>Number of Bytes written to the file</returns>
        override public int writeRecordToFile(StreamWriter file, ExtractOutputType_e outputType)
        {
            string outp;
            if (outputType == ExtractOutputType_e.csv)
            {
                outp = DataLogFileName;
                outp += "," + VersionNumber.ToString();
                outp += "," + TimeStampSec.ToString();
                outp += "," + LogDateTime.ToString();
            }
            else
            {
                outp = "Json String";

            }
            file.WriteLine(outp);
            return outp.Length;
        }

        /// <summary>
        /// Write CSV Column Names as a string of comma-seperated names to the given file.
        /// </summary>
        /// <param name="file"></param>
        /// <returns>Size of string written. A negative value indicates an error.</returns>
        override public int writeColumnNames(StreamWriter file, ExtractOutputType_e outputType)
        {
            int noBytes = 0;
            if (outputType == ExtractOutputType_e.csv)
            {
                string columnNames = "DataLogFileName";
                columnNames += "," + "VersionNumber";
                columnNames += "," + "TimeStampSec";
                columnNames += "," + "LogDateTime";
                noBytes = columnNames.Length;
                file.WriteLine(columnNames);
            }
            return noBytes;
        }

    }
}
