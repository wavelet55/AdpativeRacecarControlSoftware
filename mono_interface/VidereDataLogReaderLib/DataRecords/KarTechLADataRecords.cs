using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using VidereDataLogReaderLib.Utils;

namespace VidereDataLogReaderLib.DataRecords
{
    public class KarTechLACommandDataRecord : DataLogRecord
    {
        public byte ClutchEnable;
        public byte MotorEnable;
        public byte ManualExtControl;
        public double PositionPercent;
        public double SetPositionInches;

        //TimeStamp in seconds... all records
        //have a timestamp associated with the recorded data.
        //Within the system the timestamps must be synchronous so that
        //recorded data can be ordered by time.
        public double TimeStampSec;

        //Private members that are not serialized.
        private const int MaxLogRecordSize = 32;
        private byte[] byteArray;
        private ByteArrayReader br;

        private DataRecordType_e _dataRecordType = DataRecordType_e.KTLA_Brake_Cmd;

        public KarTechLACommandDataRecord()
        {
            byteArray = new byte[MaxLogRecordSize];
            br = new ByteArrayReader(byteArray, Utils.EndianOrder_e.big);
            Clear();
        }

        override public DataRecordType_e GetDataRecordType()
        {
            return _dataRecordType;
        }

        public void SetBrakeOrThrottleType(bool brakeType)
        {
            _dataRecordType = brakeType ? DataRecordType_e.KTLA_Brake_Cmd : DataRecordType_e.KTLA_Throttle_Cmd;
        }

        override public void Clear()
        {
            ClutchEnable = 0;
            MotorEnable = 0;
            ManualExtControl = 0;
            PositionPercent = 0;
            SetPositionInches = 0;
            TimeStampSec = 0;
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
                TimeStampSec = br.readDouble();
                ClutchEnable = br.readUInt8();
                MotorEnable = br.readUInt8();
                ManualExtControl = br.readUInt8();
                PositionPercent = br.readDouble();
                SetPositionInches = br.readDouble();
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
                outp = TimeStampSec.ToString();
                outp += "," + ClutchEnable.ToString();
                outp += "," + MotorEnable.ToString();
                outp += "," + ManualExtControl.ToString();
                outp += "," + PositionPercent.ToString();
                outp += "," + SetPositionInches.ToString();
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
                string columnNames = "TimeStampSec";
                columnNames += "," + "ClutchEnable";
                columnNames += "," + "MotorEnable";
                columnNames += "," + "ManualExtControl";
                columnNames += "," + "PositionPercent";
                columnNames += "," + "SetPositionInches";
                file.WriteLine(columnNames);
                noBytes = columnNames.Length;
            }
            return noBytes;
        }

    }

    public class KarTechLAStatusDataRecord : DataLogRecord
    {

        public double PositionPercent;
        public double ActuatorPostionInches;
        public double MotorCurrentAmps;

        //TimeStamp in seconds... all records
        //have a timestamp associated with the recorded data.
        //Within the system the timestamps must be synchronous so that
        //recorded data can be ordered by time.
        public double TimeStampSec;

        //Private members that are not serialized.
        private const int MaxLogRecordSize = 40;
        private byte[] byteArray;
        private ByteArrayReader br;

        private DataRecordType_e _recordType = DataRecordType_e.KTLA_Brake_Status;

        public KarTechLAStatusDataRecord()
        {
            byteArray = new byte[MaxLogRecordSize];
            br = new ByteArrayReader(byteArray, Utils.EndianOrder_e.big);
            Clear();
        }

        override public DataRecordType_e GetDataRecordType()
        {
            return _recordType;
        }

        public void SetBrakeOrThrottleType(bool brakeType)
        {
            _recordType = brakeType ? DataRecordType_e.KTLA_Brake_Status : DataRecordType_e.KTLA_Throttle_Status;
        }

        override public void Clear()
        {
            PositionPercent = 0;
            ActuatorPostionInches = 0;
            MotorCurrentAmps = 0;
            TimeStampSec = 0;
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
                TimeStampSec = br.readDouble();
                PositionPercent = br.readDouble();
                ActuatorPostionInches = br.readDouble();
                MotorCurrentAmps = br.readDouble();
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
                outp = TimeStampSec.ToString();
                outp += "," + PositionPercent.ToString();
                outp += "," + ActuatorPostionInches.ToString();
                outp += "," + MotorCurrentAmps.ToString();
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
                string columnNames = "TimeStampSec";
                columnNames += "," + "PositionPercent";
                columnNames += "," + "ActuatorPostionInches";
                columnNames += "," + "MotorCurrentAmps";
                file.WriteLine(columnNames);
                noBytes = columnNames.Length;
            }
            return noBytes;
        }

    }

}

