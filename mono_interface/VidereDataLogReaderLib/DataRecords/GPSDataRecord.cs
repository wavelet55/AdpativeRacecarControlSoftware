using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using VidereDataLogReaderLib.Utils;

namespace VidereDataLogReaderLib.DataRecords
{
    public class GPSDataRecord : DataLogRecord
    {
        public double GPSTimeStampSec;
        public double LatitudeDeg;
        public double LongitudeDeg;
        public double X_PosMeters;      //X-Axis is the East-West Axis with East postive
        public double Y_PosMeters;      //Y-Axis is the North-South Axis with North Positive
        public double Z_PosMeters;      //Meters above mean sea level
        public double X_Velocity_mps;   //Velocity in Meters per second
        public double Y_Velocity_mps;   //Velocity in Meters per second

        //public double AltitudeM;        //Meters above mean sea level
        //public double SpeedMPS;         //Meters per Second
        //public double TravelAngleDeg;   //Velocity angle relative to North
        //public double Dilution;         //Redundant... see Horiz and Vert Dilution
        public double HorzDilution;
        public double VertDilution;
        public UInt32 TrackingSatellites;
        public byte FixType;
        public byte FixQuality;
        public byte FixStatus;

        //TimeStamp in seconds... all records
        //have a timestamp associated with the recorded data.
        //Within the system the timestamps must be synchronous so that
        //recorded data can be ordered by time.
        public double TimeStampSec;

        //Private members that are not serialized.
        private const int MaxLogRecordSize = 128;
        private byte[] byteArray;
        private ByteArrayReader br;

        public GPSDataRecord()
        {
            byteArray = new byte[MaxLogRecordSize];
            br = new ByteArrayReader(byteArray, Utils.EndianOrder_e.big);
            Clear();
        }

        override public DataRecordType_e GetDataRecordType()
        {
            return DataRecordType_e.GPSRT_GPS_Data;
        }


        override public void Clear()
        {
            GPSTimeStampSec = 0;
            LatitudeDeg = 0;
            LongitudeDeg = 0;
            X_PosMeters = 0;
            Y_PosMeters = 0;
            Z_PosMeters = 0;
            X_Velocity_mps = 0;
            Y_Velocity_mps = 0;
            HorzDilution = 0;
            VertDilution = 0;
            TrackingSatellites = 0;
            FixType = 0;
            FixQuality = 0;
            FixStatus = 0;
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
                GPSTimeStampSec = br.readDouble();
                LatitudeDeg = br.readDouble();
                LongitudeDeg = br.readDouble();

                X_PosMeters = br.readDouble();
                Y_PosMeters = br.readDouble();
                Z_PosMeters = br.readDouble();
                X_Velocity_mps = br.readDouble();
                Y_Velocity_mps = br.readDouble();

                HorzDilution = br.readDouble();
                VertDilution = br.readDouble();
                TrackingSatellites = br.readUInt32();
                FixType = br.readUInt8();
                FixQuality = br.readUInt8();
                FixStatus = br.readUInt8();
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
                outp += "," + GPSTimeStampSec.ToString();
                outp += "," + LatitudeDeg.ToString();
                outp += "," + LongitudeDeg.ToString();
                outp += "," + X_PosMeters.ToString();
                outp += "," + Y_PosMeters.ToString();
                outp += "," + Z_PosMeters.ToString();
                outp += "," + X_Velocity_mps.ToString();
                outp += "," + Y_Velocity_mps.ToString();
                outp += "," + HorzDilution.ToString();
                outp += "," + VertDilution.ToString();
                outp += "," + TrackingSatellites.ToString();
                outp += "," + FixType.ToString();
                outp += "," + FixQuality.ToString();
                outp += "," + FixStatus.ToString();
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
                columnNames += "," + "GPSTimeStampSec";
                columnNames += "," + "LatitudeDeg";
                columnNames += "," + "LongitudeDeg";
                columnNames += "," + "X_PosMeters";
                columnNames += "," + "Y_PosMeters";
                columnNames += "," + "Z_PosMeters";
                columnNames += "," + "X_Velocity_mps";
                columnNames += "," + "Y_Velocity_mps";
                columnNames += "," + "HorzDilution";
                columnNames += "," + "VertDilution";
                columnNames += "," + "TrackingSatellites";
                columnNames += "," + "FixType";
                columnNames += "," + "FixQuality";
                columnNames += "," + "FixStatus";
                file.WriteLine(columnNames);
                noBytes = columnNames.Length;
            }
            return noBytes;
        }

    }
}



