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

namespace VidereDataLogReaderLib.DataRecords
{

    /// <summary>
    /// The data record types must match the Videre Data Record Type.
    /// </summary>
    public enum DataRecordType_e
    {
        stdHeader,
        IMURT_AccelGyro,
        IMURT_HeadOrientation,
        SPRT_SipnPuffVals,
        GPSRT_GPS_Header,
        GPSRT_GPS_Data,
        KTLA_Brake_Cmd,
        KTLA_Brake_Status,
        KTLA_Throttle_Cmd,
        KTLA_Throttle_Status,
        EPAS_Steering_Cmd,
        EPAS_Steering_Status,
        TrackHeadOrientation,    //Head orientation based upon Image Processing.
        SystemControl
    }

    /// <summary>
    /// Output file format types.
    /// </summary>
    public enum ExtractOutputType_e
    {
        csv,        //Comma Seperated Values
        json        //Jason Format
    }

    //Must be defined the smame as in Videre.
    public enum EndianOrder_e
    {
        LittleEndian,
        BigEndian
    }


    /// <summary>
    /// All data records must be derived from this class.
    /// </summary>
    public abstract class DataLogRecord
    {

        public DataLogRecord() { }

        /// <summary>
        /// Each record must return its record type
        /// </summary>
        /// <returns></returns>
        abstract public DataRecordType_e GetDataRecordType();

        protected int readBufOfDataFromFile(FileStream file, byte[] buf, int noBytesToRead)
        {
            int numBytesRead = 0;
            noBytesToRead = noBytesToRead > buf.Length ? buf.Length : noBytesToRead;
            try
            {
                numBytesRead = file.Read(buf, 0, noBytesToRead);
            }
            catch
            {
                numBytesRead = -1;
            }
            if (numBytesRead < noBytesToRead)
            {
                return 0;
            }
            return numBytesRead;
        }

        /// <summary>
        /// Clear the record variables.
        /// </summary>
        abstract public void Clear();

        /// <summary>
        /// Read and deserialize record from the file.
        /// The record type must already be established as the 
        /// concrete object must be of the record type in the file
        /// </summary>
        /// <param name="file">The filestreem that is already at the location of the record.</param>
        /// <param name="recordSize"></param>
        /// <returns></returns>
        abstract public int readRecordFromFile(FileStream file, int recordSize);

        /// <summary>
        /// Write this record out to the output file in the given format.
        /// </summary>
        /// <param name="file">File steam set to the location of the next record to be written.</param>
        /// <param name="outputType"></param>
        /// <returns>Number of Bytes written to the file. A negative value indicates an error.</returns>
        abstract public int writeRecordToFile(StreamWriter file, ExtractOutputType_e outputType);

        /// <summary>
        /// Write CSV Column Names as a string of comma-seperated names to the given file.
        /// </summary>
        /// <param name="file"></param>
        /// <returns>Size of string written. A negative value indicates an error.</returns>
        abstract public int writeColumnNames(StreamWriter file, ExtractOutputType_e outputType);


        static public DataLogRecord createNewRecord(DataRecordType_e recordType)
        {
            DataLogRecord dataRecord = null;
            switch(recordType)
            {
                case DataRecordType_e.stdHeader:
                    dataRecord = new StdHeaderDataRecord();
                    break;
                case DataRecordType_e.IMURT_AccelGyro:
                    dataRecord = new IMU_AccelGyroRecord();
                    break;
                case DataRecordType_e.IMURT_HeadOrientation:
                    dataRecord = new IMU_HeadOrientationRecord();
                    break;
                case DataRecordType_e.SPRT_SipnPuffVals:
                    dataRecord = new SipnPuffDataRecord();
                    break;
                case DataRecordType_e.GPSRT_GPS_Data:
                    dataRecord = new GPSDataRecord();
                    break;
                case DataRecordType_e.GPSRT_GPS_Header:
                    dataRecord = new GPSDataRecordHeader();
                    break;
                case DataRecordType_e.KTLA_Brake_Cmd:
                    dataRecord = new KarTechLACommandDataRecord();
                    ((KarTechLACommandDataRecord)dataRecord).SetBrakeOrThrottleType(true);
                    break;
                case DataRecordType_e.KTLA_Brake_Status:
                    dataRecord = new KarTechLAStatusDataRecord();
                    ((KarTechLAStatusDataRecord)dataRecord).SetBrakeOrThrottleType(true);
                    break;
                case DataRecordType_e.KTLA_Throttle_Cmd:
                    dataRecord = new KarTechLACommandDataRecord();
                    ((KarTechLACommandDataRecord)dataRecord).SetBrakeOrThrottleType(false);
                    break;
                case DataRecordType_e.KTLA_Throttle_Status:
                    dataRecord = new KarTechLAStatusDataRecord();
                    ((KarTechLAStatusDataRecord)dataRecord).SetBrakeOrThrottleType(false);
                    break;
                case DataRecordType_e.EPAS_Steering_Cmd:
                    dataRecord = new EpasSteeringCommandDataRecord();
                    break;
                case DataRecordType_e.EPAS_Steering_Status:
                    dataRecord = new EpasSteeringStatusDataRecord();
                    break;
                case DataRecordType_e.TrackHeadOrientation:
                    dataRecord = new TrackHeadDataRecord();
                    break;
                case DataRecordType_e.SystemControl:
                    dataRecord = new VidereSystemControlDataRecord();
                    break;
            }
            return dataRecord;
        }
    }
}
