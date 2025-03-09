using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using VidereDataLogReaderLib.Utils;

namespace VidereDataLogReaderLib.DataRecords
{
    public class EpasSteeringCommandDataRecord : DataLogRecord
    {
        public byte SteeringControlEnabled;
        public byte ManualExtControl;
        public byte SteeringTorqueMap;
        public double SteeringTorquePercent;

        //TimeStamp in seconds... all records
        //have a timestamp associated with the recorded data.
        //Within the system the timestamps must be synchronous so that
        //recorded data can be ordered by time.
        public double TimeStampSec;

        //Private members that are not serialized.
        private const int MaxLogRecordSize = 32;
        private byte[] byteArray;
        private ByteArrayReader br;

        public EpasSteeringCommandDataRecord()
        {
            byteArray = new byte[MaxLogRecordSize];
            br = new ByteArrayReader(byteArray, Utils.EndianOrder_e.big);
            Clear();
        }

        override public DataRecordType_e GetDataRecordType()
        {
            return DataRecordType_e.EPAS_Steering_Cmd;
        }

        override public void Clear()
        {
            SteeringControlEnabled = 0;
            ManualExtControl = 0;
            SteeringTorqueMap = 0;
            SteeringTorquePercent = 0;
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
                SteeringControlEnabled = br.readUInt8();
                ManualExtControl = br.readUInt8();
                SteeringTorqueMap = br.readUInt8();
                SteeringTorquePercent = br.readDouble();
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
                outp += "," + SteeringControlEnabled.ToString();
                outp += "," + ManualExtControl.ToString();
                outp += "," + SteeringTorqueMap.ToString();
                outp += "," + SteeringTorquePercent.ToString();
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
                columnNames += "," + "SteeringControlEnabled";
                columnNames += "," + "ManualExtControl";
                columnNames += "," + "SteeringTorqueMap";
                columnNames += "," + "SteeringTorquePercent";
                file.WriteLine(columnNames);
                noBytes = columnNames.Length;
            }
            return noBytes;
        }

    }

    public class EpasSteeringStatusDataRecord : DataLogRecord
    {

        public double MotorCurrentAmps;
        public double PWMDutyCyclePercent;
        public double MotorTorquePercent;
        public double SupplyVoltage;
        public double TempDegC;
        public double SteeringAngleDeg;
        public byte SteeringTorqueMap;
        public byte SwitchPosition;
        public byte TorqueA;
        public byte TorqueB;
        public byte ErrorCode;
        public byte StatusFlags;
        public byte LimitFlags;

        //TimeStamp in seconds... all records
        //have a timestamp associated with the recorded data.
        //Within the system the timestamps must be synchronous so that
        //recorded data can be ordered by time.
        public double TimeStampSec;

        //Private members that are not serialized.
        private const int MaxLogRecordSize = 80;
        private byte[] byteArray;
        private ByteArrayReader br;

        public EpasSteeringStatusDataRecord()
        {
            byteArray = new byte[MaxLogRecordSize];
            br = new ByteArrayReader(byteArray, Utils.EndianOrder_e.big);
            Clear();
        }

        override public DataRecordType_e GetDataRecordType()
        {
            return DataRecordType_e.EPAS_Steering_Status;
        }

        override public void Clear()
        {
            MotorCurrentAmps = 0;
            PWMDutyCyclePercent = 0;
            MotorTorquePercent = 0;
            SupplyVoltage = 0;
            TempDegC = 0;
            SteeringAngleDeg = 0;
            SteeringTorqueMap = 0;
            SwitchPosition = 0;
            TorqueA = 0;
            TorqueB = 0;
            ErrorCode = 0;
            StatusFlags = 0;
            LimitFlags = 0;
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
                MotorCurrentAmps = br.readDouble();
                PWMDutyCyclePercent = br.readDouble();
                MotorTorquePercent = br.readDouble();
                SupplyVoltage = br.readDouble();
                TempDegC = br.readDouble();
                SteeringAngleDeg = br.readDouble();
                SteeringTorqueMap = br.readUInt8();
                SwitchPosition = br.readUInt8();
                TorqueA = br.readUInt8();
                TorqueB = br.readUInt8();
                ErrorCode = br.readUInt8();
                StatusFlags = br.readUInt8();
                LimitFlags = br.readUInt8();
                TimeStampSec = br.readUInt8();
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
                outp += "," + MotorCurrentAmps.ToString();
                outp += "," + PWMDutyCyclePercent.ToString();
                outp += "," + MotorTorquePercent.ToString();
                outp += "," + SupplyVoltage.ToString();
                outp += "," + TempDegC.ToString();
                outp += "," + SteeringAngleDeg.ToString();
                outp += "," + SteeringTorqueMap.ToString();
                outp += "," + SwitchPosition.ToString();
                outp += "," + TorqueA.ToString();
                outp += "," + TorqueB.ToString();
                outp += "," + ErrorCode.ToString();
                outp += "," + StatusFlags.ToString();
                outp += "," + LimitFlags.ToString();
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
                columnNames += "," + "MotorCurrentAmps";
                columnNames += "," + "PWMDutyCyclePercent";
                columnNames += "," + "MotorTorquePercent";
                columnNames += "," + "SupplyVoltage";
                columnNames += "," + "TempDegC";
                columnNames += "," + "SteeringAngleDeg";
                columnNames += "," + "SteeringTorqueMap";
                columnNames += "," + "SwitchPosition";
                columnNames += "," + "TorqueA";
                columnNames += "," + "TorqueB";
                columnNames += "," + "ErrorCode";
                columnNames += "," + "StatusFlags";
                columnNames += "," + "LimitFlags";
                file.WriteLine(columnNames);
                noBytes = columnNames.Length;
            }
            return noBytes;
        }

    }

}

