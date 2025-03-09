using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using VidereDataLogReaderLib.Utils;

namespace VidereDataLogReaderLib.DataRecords
{
    public enum IMULocation_e
    {
        Fixed,
        Head,
        Both,
        NA
    }

    public class IMU_HeadOrientationRecord : DataLogRecord
    {

        //TimeStamp in seconds... all records
        //have a timestamp associated with the recorded data.
        //Within the system the timestamps must be synchronous so that
        //recorded data can be ordered by time.
        public double TimeStampSec;

        public double IMUTimeStampSec;

        public double RollAngleRad;
        public double PitchAngleRad;
        public double YawAngleRad;

        //Private members that are not serialized.
        private const int MaxLogRecordSize = 64;
        private byte[] byteArray;
        private ByteArrayReader br;

        public IMU_HeadOrientationRecord()
        {
            byteArray = new byte[MaxLogRecordSize];
            br = new ByteArrayReader(byteArray, Utils.EndianOrder_e.big);
            Clear();
        }

        override public DataRecordType_e GetDataRecordType()
        {
            return DataRecordType_e.IMURT_HeadOrientation;
        }


        override public void Clear()
        {
            TimeStampSec = 0;
            IMUTimeStampSec = 0;
            RollAngleRad = 0;
            PitchAngleRad = 0;
            YawAngleRad = 0;
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
                IMUTimeStampSec = br.readDouble();
                RollAngleRad = br.readDouble();
                PitchAngleRad = br.readDouble();
                YawAngleRad = br.readDouble();
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
                outp += "," + IMUTimeStampSec.ToString();
                outp += "," + RollAngleRad.ToString();
                outp += "," + PitchAngleRad.ToString();
                outp += "," + YawAngleRad.ToString();
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
                columnNames += "," + "IMUTimeStampSec";
                columnNames += "," + "RollAngleRad";
                columnNames += "," + "PitchAngleRad";
                columnNames += "," + "YawAngleRad";
                file.WriteLine(columnNames);
                noBytes = columnNames.Length;
            }
            return noBytes;
        }

    }

    public class IMU_AccelGyroRecord : DataLogRecord
    {

        //TimeStamp in seconds... all records
        //have a timestamp associated with the recorded data.
        //Within the system the timestamps must be synchronous so that
        //recorded data can be ordered by time.
        public double TimeStampSec;

        public double IMUTimeStampSec;

        public IMULocation_e IMULocation;

        public double AcceleratorMPS2_x;
        public double AcceleratorMPS2_y;
        public double AcceleratorMPS2_z;
        public double GyroRadPerSec_x;
        public double GyroRadPerSec_y;
        public double GyroRadPerSec_z;

        //Private members that are not serialized.
        private const int MaxLogRecordSize = 128;
        private byte[] byteArray;
        private ByteArrayReader br;

        public IMU_AccelGyroRecord()
        {
            byteArray = new byte[MaxLogRecordSize];
            br = new ByteArrayReader(byteArray, Utils.EndianOrder_e.big);
            Clear();
        }

        override public DataRecordType_e GetDataRecordType()
        {
            return DataRecordType_e.IMURT_AccelGyro;
        }


        override public void Clear()
        {
            TimeStampSec = 0;
            IMUTimeStampSec = 0;
            IMULocation = IMULocation_e.Fixed;
            AcceleratorMPS2_x = 0;
            AcceleratorMPS2_y = 0;
            AcceleratorMPS2_z = 0;
            GyroRadPerSec_x = 0;
            GyroRadPerSec_y = 0;
            GyroRadPerSec_z = 0;
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
                IMULocation = (IMULocation_e)br.readUInt8();
                TimeStampSec = br.readDouble();
                IMUTimeStampSec = br.readDouble();
                AcceleratorMPS2_x = br.readDouble();
                AcceleratorMPS2_y = br.readDouble();
                AcceleratorMPS2_z = br.readDouble();
                GyroRadPerSec_x = br.readDouble();
                GyroRadPerSec_y = br.readDouble();
                GyroRadPerSec_z = br.readDouble();
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
                outp = IMULocation.ToString();
                outp += "," + TimeStampSec.ToString();
                outp += "," + IMUTimeStampSec.ToString();
                outp += "," + AcceleratorMPS2_x.ToString();
                outp += "," + AcceleratorMPS2_y.ToString();
                outp += "," + AcceleratorMPS2_z.ToString();
                outp += "," + GyroRadPerSec_x.ToString();
                outp += "," + GyroRadPerSec_y.ToString();
                outp += "," + GyroRadPerSec_z.ToString();
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
                string columnNames = "IMULocation";
                columnNames += "," + "TimeStampSec";
                columnNames += "," + "IMUTimeStampSec";
                columnNames += "," + "AcceleratorMPS2_x";
                columnNames += "," + "AcceleratorMPS2_y";
                columnNames += "," + "AcceleratorMPS2_z";
                columnNames += "," + "GyroRadPerSec_x";
                columnNames += "," + "GyroRadPerSec_y";
                columnNames += "," + "GyroRadPerSec_z";
                noBytes = columnNames.Length;
                file.WriteLine(columnNames);
            }
            return noBytes;
        }

    }


}

