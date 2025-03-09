using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using VidereDataLogReaderLib.Utils;

namespace VidereDataLogReaderLib.DataRecords
{
    public class VidereSystemControlDataRecord : DataLogRecord
    {
        public byte ControlTypeState;
        public byte DriverEnableSW;
        public byte DriverTorqueHit;
         
        public double deltaTime;
        public double SipnPuffValue;
        public double ThrottleBrakeIntegralVal;
        public double ThrottleControlVal;
        public double BrakecontrolVal;
        public double HeadRollAngleDegrees;    //Postive angle is roll head to right
        public double HeadPitchAngleDegrees;   //Positive angle is head up
        public double HeadYawAngleDegrees;     //Positive angle is head to right
        public double HeadLRAngleClamped;
        public double HeadLRAngleLPF;
        public double SteeringAngle;
        public double SAError;
        public double DtSAError;
        public double IntgSAError;
        public double SteeringTorqueCtrl;
        public double VehiclePos_X;
        public double VehiclePos_Y;
        public double VehicleSpeed;
        public double VehicleVel_X;
        public double VehicleVel_Y;


        //TimeStamp in seconds... all records
        //have a timestamp associated with the recorded data.
        //Within the system the timestamps must be synchronous so that
        //recorded data can be ordered by time.
        public double TimeStampSec;

        //Private members that are not serialized.
        private const int MaxLogRecordSize = 200;
        private byte[] byteArray;
        private ByteArrayReader br;

        public VidereSystemControlDataRecord()
        {
            byteArray = new byte[MaxLogRecordSize];
            br = new ByteArrayReader(byteArray, Utils.EndianOrder_e.big);
            Clear();
        }

        override public DataRecordType_e GetDataRecordType()
        {
            return DataRecordType_e.SystemControl;
        }


        override public void Clear()
        {
            ControlTypeState = 0;
            DriverEnableSW = 0;
            DriverTorqueHit = 0;
            deltaTime = 0;
            SipnPuffValue = 0;
            ThrottleBrakeIntegralVal = 0;
            ThrottleControlVal = 0;
            BrakecontrolVal = 0;
            HeadRollAngleDegrees = 0;   
            HeadPitchAngleDegrees = 0;
            HeadYawAngleDegrees = 0;
            HeadLRAngleClamped = 0;
            HeadLRAngleLPF = 0;
            SteeringAngle = 0;
            SAError = 0;
            DtSAError = 0;
            IntgSAError = 0;
            SteeringTorqueCtrl = 0;
            VehiclePos_X = 0;
            VehiclePos_Y = 0;
            VehicleSpeed = 0;
            VehicleVel_X = 0;
            VehicleVel_Y = 0;
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
                ControlTypeState = br.readUInt8();
                DriverEnableSW = br.readUInt8();
                DriverTorqueHit = br.readUInt8();

                deltaTime = br.readDouble();
                SipnPuffValue = br.readDouble();
                ThrottleBrakeIntegralVal = br.readDouble();
                ThrottleControlVal = br.readDouble();
                BrakecontrolVal = br.readDouble();

                HeadRollAngleDegrees = br.readDouble();
                HeadPitchAngleDegrees = br.readDouble();
                HeadYawAngleDegrees = br.readDouble();
                HeadLRAngleClamped = br.readDouble();
                HeadLRAngleLPF = br.readDouble();

                SteeringAngle = br.readDouble();
                SAError = br.readDouble();
                DtSAError = br.readDouble();
                IntgSAError = br.readDouble();
                SteeringTorqueCtrl = br.readDouble();

                VehiclePos_X = br.readDouble();
                VehiclePos_Y = br.readDouble();
                VehicleSpeed = br.readDouble();
                VehicleVel_X = br.readDouble();
                VehicleVel_Y = br.readDouble();
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
                outp += "," + ControlTypeState.ToString();
                outp += "," + DriverEnableSW.ToString();
                outp += "," + DriverTorqueHit.ToString();
                outp += "," + deltaTime.ToString();
                outp += "," + SipnPuffValue.ToString();
                outp += "," + ThrottleBrakeIntegralVal.ToString();
                outp += "," + ThrottleControlVal.ToString();
                outp += "," + BrakecontrolVal.ToString();
                outp += "," + HeadRollAngleDegrees.ToString();
                outp += "," + HeadPitchAngleDegrees.ToString();
                outp += "," + HeadYawAngleDegrees.ToString();
                outp += "," + HeadLRAngleClamped.ToString();
                outp += "," + HeadLRAngleLPF.ToString();
                outp += "," + SteeringAngle.ToString();
                outp += "," + SAError.ToString();
                outp += "," + DtSAError.ToString();
                outp += "," + IntgSAError.ToString();
                outp += "," + SteeringTorqueCtrl.ToString();
                outp += "," + VehiclePos_X.ToString();
                outp += "," + VehiclePos_Y.ToString();
                outp += "," + VehicleSpeed.ToString();
                outp += "," + VehicleVel_X.ToString();
                outp += "," + VehicleVel_Y.ToString();
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
                columnNames += "," + "ControlTypeState";
                columnNames += "," + "DriverEnableSW";
                columnNames += "," + "DriverTorqueHit";
                columnNames += "," + "deltaTime";
                columnNames += "," + "SipnPuffValue";
                columnNames += "," + "ThrottleBrakeIntegralVal";
                columnNames += "," + "ThrottleControlVal";
                columnNames += "," + "BrakecontrolVal";
                columnNames += "," + "HeadRollAngleDegrees";
                columnNames += "," + "HeadPitchAngleDegrees";
                columnNames += "," + "HeadYawAngleDegrees";
                columnNames += "," + "HeadLRAngleClamped";
                columnNames += "," + "HeadLRAngleLPF";
                columnNames += "," + "SteeringAngle";
                columnNames += "," + "SAError";
                columnNames += "," + "DtSAError";
                columnNames += "," + "IntgSAError";
                columnNames += "," + "SteeringTorqueCtrl";
                columnNames += "," + "VehiclePos_X";
                columnNames += "," + "VehiclePos_Y";
                columnNames += "," + "VehicleSpeed";
                columnNames += "," + "VehicleVel_X";
                columnNames += "," + "VehicleVel_Y";

                file.WriteLine(columnNames);
                noBytes = columnNames.Length;
            }
            return noBytes;
        }

    }
}


