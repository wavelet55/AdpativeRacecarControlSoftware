using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using VidereDataLogReaderLib.Utils;

namespace VidereDataLogReaderLib.DataRecords
{
    public class TrackHeadDataRecord : DataLogRecord
    {
        public bool IsDataValid;

        //Rodrigus Orientation Vector
        public double HeadOrientationQuaternion_W;
        public double HeadOrientationQuaternion_X;
        public double HeadOrientationQuaternion_Y;
        public double HeadOrientationQuaternion_Z;

        public double HeadTranslationVec_X;
        public double HeadTranslationVec_Y;
        public double HeadTranslationVec_Z;

        //TimeStamp in seconds... all records
        //have a timestamp associated with the recorded data.
        //Within the system the timestamps must be synchronous so that
        //recorded data can be ordered by time.
        public double TimeStampSec;

        //Time the Image was captures... The difference between
        //this time and TimeStampSec is the amount of time it took 
        //to process the image and the the head orientation info.
        public double ImageCaptureTimeStampSec;

        UInt32 ImageNumber;

        //Private members that are not serialized.
        private const int MaxLogRecordSize = 96;
        private byte[] byteArray;
        private ByteArrayReader br;

        public TrackHeadDataRecord()
        {
            byteArray = new byte[MaxLogRecordSize];
            br = new ByteArrayReader(byteArray, Utils.EndianOrder_e.big);
            Clear();
        }

        override public DataRecordType_e GetDataRecordType()
        {
            return DataRecordType_e.TrackHeadOrientation;
        }


        override public void Clear()
        {
            IsDataValid = false;
            HeadOrientationQuaternion_W = 0;
            HeadOrientationQuaternion_X = 0;
            HeadOrientationQuaternion_Y = 0;
            HeadOrientationQuaternion_Z = 0;
            HeadTranslationVec_X = 0;
            HeadTranslationVec_Y = 0;
            HeadTranslationVec_Z = 0;
            TimeStampSec = 0;
            ImageCaptureTimeStampSec = 0;
            ImageNumber = 0;
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
                ImageCaptureTimeStampSec = br.readDouble();
                ImageNumber = br.readUInt32();
                IsDataValid = br.readUInt8() == 0 ? false : true;
                HeadOrientationQuaternion_W = br.readDouble();
                HeadOrientationQuaternion_X = br.readDouble();
                HeadOrientationQuaternion_Y = br.readDouble();
                HeadOrientationQuaternion_Z = br.readDouble();
                HeadTranslationVec_X = br.readDouble();
                HeadTranslationVec_Y = br.readDouble();
                HeadTranslationVec_Z = br.readDouble();
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
                outp += "," + ImageCaptureTimeStampSec.ToString();
                outp += "," + ImageNumber.ToString();
                outp += "," + (IsDataValid ? '1' : '0');
                outp += "," + HeadOrientationQuaternion_W.ToString();
                outp += "," + HeadOrientationQuaternion_X.ToString();
                outp += "," + HeadOrientationQuaternion_Y.ToString();
                outp += "," + HeadOrientationQuaternion_Z.ToString();
                outp += "," + HeadTranslationVec_X.ToString();
                outp += "," + HeadTranslationVec_Y.ToString();
                outp += "," + HeadTranslationVec_Z.ToString();
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
                columnNames += "," + "ImageCaptureTimeStampSec";
                columnNames += "," + "ImageNumber";
                columnNames += "," + "IsDataValid";
                columnNames += "," + "Q_W";
                columnNames += "," + "Q_X";
                columnNames += "," + "Q_Y";
                columnNames += "," + "Q_Z";
                columnNames += "," + "TVec_X";
                columnNames += "," + "TVec_Y";
                columnNames += "," + "TVec_Z";
                file.WriteLine(columnNames);
                noBytes = columnNames.Length;
            }
            return noBytes;
        }

    }
}

