/* ****************************************************************
 * File: ImageExtractor.cs
 * Athr: Harry Direen
 * Date: Nov. 9, 2010
 * 
 * Academy Center for UAS Research 
 * Department of Electrical and Computer Engineering 
 * HQ USAFA/DFEC 
 * 2354 Fairchild Drive 
 * USAF Academy, CO 80840-6236 
 * 
 * Desc: Image ImageFileExtractor
 * This class extracts images with their metadata (vehical position, 
 * camera orientation, time, etc) from a single file into individual files.
 * Each image will have its own file and metadata xml file.
 *******************************************************************/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Xml;
using System.Xml.XPath;
using System.Xml.Serialization;
using System.Windows.Forms;
using HopsCommon;
using HopsCommon.Utils;
using UAS.Vehicles;

namespace ImageFileExtractor
{
    public enum ImageSensorMode_e
    {
        normal = 0,
        MJPEG = 1,      //Motion JPEG
        SampleGrabber = 2,
        DSHOW = 3,
        YUV,            //raw yuv format... must be converted to .bmp
                        //for viewing.
   }


    /// <summary>
    /// Image File Header Structure: (Big Endian Order)
    /// Image File Version:  string, 32 bytes
    /// File Header Size: int32
    /// Image header size: int32
    /// ImageSensorMode: int32
    /// ImageFileExt: string, 8 bytes
    /// VehicleId: int32
    /// VehicleType: int32
    /// VehicleSubtype: int32
    /// TailNumber: string, 32 bytes
    /// AtcCallSign: string, 32 bytes
    /// MissionId: string, 40 bytes
    /// </summary>
    public class ImageFileHeader
    {
        //Data Captured from the file header section.
        public string ImageFileVersion;
        public ImageSensorMode_e ImageSensorMode;
        public string ImageFileExt;
        public Int32 VehicleId;
        public Int32 VehicleType;
        public Int32 VehicleSubtype;
        public string TailNumber;
        public string AtcCallSign;
        public string MissionId;

        public latLon_t CoordSysCenter;
        public xyCoord_t LatLonToXYScaleFactors;

        public static string getImageFileExt(ImageSensorMode_e mode)
        {
            string ext = null;
            switch (mode)
            {
                case ImageSensorMode_e.normal:
                    ext = "mov";    //A guess I don't know for sure
                    break;
                case ImageSensorMode_e.MJPEG:
                    ext = "jpg";    //This is the standard fromat used by HOPS
                    break;
                case ImageSensorMode_e.SampleGrabber:
                    ext = "mov";    //A guess I don't know for sure
                    break;
                case ImageSensorMode_e.DSHOW:
                    ext = "ax";     //A guess I don't know for sure
                    break;

            }
            return ext;
        }
    }

    public class ImageFileHeaderReader
    {
        public const string ImageFileVersion_v1 = "ImageFileVersion=1.0";
        public const string ImageFileVersion_v1_1 = "ImageFileVersion=1.1";
        public const string ImageFileVersion_v1_2 = "ImageFileVersion=1.2";

        public string ImageFileVersion;
        private const int maxHdrSize = 2048;
        public int FileHeaderSize = 512;
        public int ImageHeaderSize = 512;

        public ImageFileHeader FileHeader;

        private byte[] headerData = new byte[maxHdrSize];
        private ByteArrayReader br;


        public ImageFileHeaderReader()
        {
            br = new ByteArrayReader(headerData, EndianOrder_e.big);
            FileHeader = new ImageFileHeader();
        }

        public void Clear()
        {
            br.Idx = 0;
        }

        public bool readFileHeaderToBuffer(FileStream fileStr)
        {
            bool error = false;
            br.Idx = 0;
            try
            {
                fileStr.Read(headerData, 0, maxHdrSize);
            }
            catch (Exception e)
            {
                MessageBox.Show("Error reading image file header, Ex: " + e.Message);
                error = true;
            }
            return error;
        }

        public void readFileVersionInfo()
        {
            ImageFileVersion = br.readString(32);   //Version is always 32 bytes.
            FileHeader.ImageFileVersion = ImageFileVersion; // JAY: So it shows up in the XML.
        }

        /// <summary>
        /// Reads the file header information.  Different 
        /// version of this routine can be created if there are different 
        /// file versions.
        /// 
        /// The file header data is the same for versions 1.0 and 1.1
        /// </summary>
        public void readFileHeaderData_v1()
        {
            FileHeaderSize = br.readInt32();
            ImageHeaderSize = br.readInt32();

            FileHeader.ImageSensorMode = (ImageSensorMode_e)br.readInt32();
            FileHeader.ImageFileExt = br.readString(8);
            FileHeader.VehicleId = br.readInt32();
            FileHeader.VehicleType = br.readInt32();
            FileHeader.VehicleSubtype = br.readInt32();
            FileHeader.TailNumber = br.readString(32);
            FileHeader.AtcCallSign = br.readString(32);
            FileHeader.MissionId = br.readString(40);
            FileHeader.CoordSysCenter.lat = br.readDouble();
            FileHeader.CoordSysCenter.lon = br.readDouble();
            FileHeader.LatLonToXYScaleFactors.x = br.readDouble();
            FileHeader.LatLonToXYScaleFactors.y = br.readDouble();
        }

        public bool readFileHeader(FileStream fileStr)
        {
            bool error = false;
            fileStr.Seek(0, SeekOrigin.Begin);       //ensure we are at the beginning of the file
            error = readFileHeaderToBuffer(fileStr);
            if (!error)
            {
                readFileVersionInfo();
                if (ImageFileHeaderReader.ImageFileVersion_v1.Equals(ImageFileVersion) ||
                    ImageFileHeaderReader.ImageFileVersion_v1_1.Equals(ImageFileVersion) ||
                    ImageFileHeaderReader.ImageFileVersion_v1_2.Equals(ImageFileVersion))
                {
                    // The file header data is the same for versions 1.0 and 1.1
                    // Only the image header data is different.
                    readFileHeaderData_v1();
                    //Assume ok and return;
                    return false; // no error
                }
                else
                {
                    MessageBox.Show("Error: Unsupported version: " + ImageFileVersion);
                    return true;
                }
            }
            return error;
        }

        public bool writeToFile(string filename)
        {
            bool error = false;
            XmlSerializer xmlSer = null;
            FileStream fs = null;

            try
            {
                fs = File.Open(filename, FileMode.Create, FileAccess.Write, FileShare.None);
                xmlSer = new XmlSerializer(FileHeader.GetType());
                xmlSer.Serialize(fs, FileHeader);
                fs.Flush();
                fs.Close();
            }
            catch (Exception e)
            {
                MessageBox.Show("Error storing image file header: " + filename + " Ex: " + e.Message);
                error = true;
            }
            return error;
        }

    }

    public class TargetInformation
    {
        /// <summary>
        /// Target Code: Hostile, Friendly, Unknown, etc.
        /// </summary>
        public ImageProcTargetCode_e TargetHostileEtcCode;

        public xyCoord_t TargetXYPos = new xyCoord_t();

        public latLon_t TargetLatLonPos = new latLon_t();

        /// <summary>
        /// Target location in the image plane
        /// </summary>
        public xyCoord_t ImgTargetLoc = new xyCoord_t();

        /// <summary>
        /// Azimuth angel in radians from the UAV to the target.
        /// </summary>
        public double Azimuth;

        /// <summary>
        /// Elevation angle in radians from the UAV to the target.
        /// </summary>
        public double Elevation;

        /// <summary>
        /// TargetHeatGradient in units???
        /// </summary>
        public double TargetHeatGradient;

        /// <summary>
        /// TargetColorValue 
        /// </summary>
        public double TargetColorValue;

        public void clear()
        {
            TargetXYPos.clear();
            TargetLatLonPos.clear();
            ImgTargetLoc.clear();
            Azimuth = 0;
            Elevation = 0;
            TargetHeatGradient = 0;
        }
    }

    public class ImageHeader
    {
        public const double DegsPerRad = 180.0 / Math.PI;

        public int ImageNumber;
        public int ImageSize;
        public DateTime DateTimeStamp;

        /// <summary>
        /// Time in seconds relative to "1970-01-01T00:00:00"
        /// </summary>
        public double TimeSec;      //Time is seconds from the start of the mission.
        public double Latitude;
        public double Longitude;
        public double AltitudeMSL;
        public double HeightAGL;
        public double AirSpeed;
        public double CompassHeading;
        public double VelNorth;
        public double VelEast;
        public double VelDown;
        public double Roll;
        public double Pitch;
        public double Yaw;
        public double CameraAzim;
        public double CameraElev;

        //Image Processing Results:
        /// <summary>
        /// ValidTarget will be > 0 if a target was found in the current image.
        /// It will be 0 if no target was found... in this case the other 
        /// parmeters in this message will be un-defined.
        /// The value will be less than zero if there was an imgage processing error
        /// or invalid image data.
        /// </summary>
        public int ValidTarget;

        /// <summary>
        /// ImageSensorType: EO/IR
        /// </summary>
        public ImageSensorType_e ImageSensorType;

        public xyCoord_t[] ImageGroundCornerPostions = new xyCoord_t[4];

        /// <summary>
        /// number of meaningful entries in targetInformation array
        /// </summary>
        public int NumTargets; 

        /// <summary>
        /// Each element of this array represents the position of a target.
        /// </summary>
        public TargetInformation[] targetInformation = new TargetInformation[4];

        public ImageHeader()
        {
            for(int indTarget = 0; indTarget<targetInformation.Length; indTarget++)
            {
                targetInformation[indTarget] = new TargetInformation(); 
            }
        }

        public void clearTgtInvalidParams()
        {
            NumTargets = 0;
            for (int indTarget = 0; indTarget < targetInformation.Length; indTarget++)
            {
                targetInformation[indTarget].clear();
            }
        }

        /// <summary>
        /// Write the first line of the CSV file, return it as a string.
        /// </summary>
        /// <param name="version">Version of the CSV file to write.</param>
        /// <returns></returns>
        public static string CSVHdrString(string version)
        {
            if (ImageFileHeaderReader.ImageFileVersion_v1.Equals(version))
            {
                return CSVHdrString_v1();
            }
            else if (ImageFileHeaderReader.ImageFileVersion_v1_1.Equals(version) ||
                ImageFileHeaderReader.ImageFileVersion_v1_2.Equals(version))
            {
                return CSVHdrString_v1_1();
            }
            else
            {
                return "Header.,This,version,not,implemented.," + version;
            }
        }

        public static string CSVHdrString_v1()
        {
            string str = "ImageNumber,ImageSize,TimeSec,LatitudeRad,LongitudeRad,PosX,PosY,AltitudeMSL,HeightAGL";
            str = string.Concat(str, ",AirSpeed,CompassHeading,VelNorth,VelEast,VelDown,RollRad,PitchRad,YawRad,CameraAzimRad,CameraElevRad");
            str = string.Concat(str, ",ValidImage,ImageSensorType");
            str = string.Concat(str, ",ImgGndCnr_0_X,ImgGndCnr_0_Y,ImgGndCnr_1_X,ImgGndCnr_1_Y,ImgGndCnr_2_X,ImgGndCnr_2_Y,ImgGndCnr_3_X,ImgGndCnr_3_Y");
            str = string.Concat(str, ",TargetType,AzimuthRad,ElevationRad,TargetXYPos_X,TargetXYPos_Y,TargetLatLonPos_LatRad,TargetLatLonPos_LonRad");
            str = string.Concat(str, ",TgtCnr_0_X,TgtCnr_0_Y,TgtCnr_1_X,TgtCnr_1_Y,TgtCnr_2_X,TgtCnr_2_Y,TgtCnr_3_X,TgtCnr_3_Y");

            str = string.Concat(str, ",ImgTargetLoc_X,ImgTargetLoc_Y");
            str = string.Concat(str, ",ImgTgtCnr_0_X,ImgTgtCnr_0_Y,ImgTgtCnr_1_X,ImgTgtCnr_1_Y,ImgTgtCnr_2_X,ImgTgtCnr_2_Y,ImgTgtCnr_3_X,ImgTgtCnr_3_Y");
            str = string.Concat(str, ",TargetHeatGradient");
            str = string.Concat(str, "\n"); //Terminate in newline
            return str;
        }

        private static string CSVHdrString_v1_1()
        {
            string str = "ImageNumber,ImageSize,TimeSec,LatitudeDeg,LongitudeDeg,PosX,PosY,AltitudeMSL,HeightAGL";
            str = string.Concat(str, ",AirSpeed,CompassHeading,VelNorth,VelEast,VelDown,RollDeg,PitchDeg,YawDeg,CameraAzimDeg,CameraElevDeg");
            str = string.Concat(str, ",ValidImage,ImageSensorType,NumTargets");
            str = string.Concat(str, ",ImgGndCnr_0_X,ImgGndCnr_0_Y,ImgGndCnr_1_X,ImgGndCnr_1_Y,ImgGndCnr_2_X,ImgGndCnr_2_Y,ImgGndCnr_3_X,ImgGndCnr_3_Y");
            str = string.Concat(str, ",Target0HostileEtc,Target0Type,Target0AzimuthDeg,Target0ElevationDeg,Target0XYPos_X,Target0XYPos_Y,Target0LatLonPos_LatDeg,Target0LatLonPos_LonDeg");
            str = string.Concat(str, ",Target0ImgLoc_X,Target0ImgLoc_Y");
            str = string.Concat(str, ",Target1HostileEtc,Target1Type,Target1AzimuthDeg,Target1ElevationDeg,Target1XYPos_X,Target1XYPos_Y,Target1LatLonPos_LatDeg,Target1LatLonPos_LonDeg");
            str = string.Concat(str, ",Target1ImgLoc_X,Target1ImgLoc_Y");
            str = string.Concat(str, ",Target2HostileEtc,Target2Type,Target2AzimuthDeg,Target2ElevationDeg,Target2XYPos_X,Target2XYPos_Y,Target2LatLonPos_LatDeg,Target2LatLonPos_LonDeg");
            str = string.Concat(str, ",Target2ImgLoc_X,Target2ImgLoc_Y");
            str = string.Concat(str, ",Target3HostileEtc,Target3Type,Target3AzimuthDeg,Target3ElevationDeg,Target3XYPos_X,Target3XYPos_Y,Target3LatLonPos_LatDeg,Target3LatLonPos_LonDeg");
            str = string.Concat(str, ",Target3ImgLoc_X,Target3ImgLoc_Y");
            str = string.Concat(str, "\n"); //Terminate in newline
            return str;
        }

        public string ToCSV_String(string version)
        {
            if (ImageFileHeaderReader.ImageFileVersion_v1.Equals(version))
            {
                return ToCSV_String_v1();
            }
            else if (ImageFileHeaderReader.ImageFileVersion_v1_1.Equals(version) ||
                ImageFileHeaderReader.ImageFileVersion_v1_2.Equals(version))
            {
                return ToCSV_String_v1_1();
            }
            else
            {
                return "Data line, This, version, not implemented.," + version + "\n";
            }
        }

        private string ToCSV_String_v1()
        {
            xyCoord_t xyPos = new xyCoord_t();
            latLon_t latLonPos = new latLon_t(Latitude, Longitude);
            if( UAVCoordinateSystem.getUAVCoordinateSystem().IsCoordinateSystemValid )
            {
                xyPos = UAVCoordinateSystem.getUAVCoordinateSystem().latLonToXY(latLonPos);
            }
            string str = string.Concat(ImageNumber.ToString(), ",", ImageSize.ToString(), ",", TimeSec.ToString());
            str = string.Concat(str, ",", Latitude.ToString(), ",", Longitude.ToString(), ",", xyPos.x.ToString(), ",", xyPos.y.ToString(), ",", AltitudeMSL.ToString(), ",", HeightAGL.ToString());
            str = string.Concat(str, ",", AirSpeed.ToString(), ",", CompassHeading.ToString(), ",", VelNorth.ToString(), ",", VelEast.ToString(), ",", VelDown.ToString());
            str = string.Concat(str, ",", Roll.ToString(), ",", Pitch.ToString(), ",", Yaw.ToString(), ",", CameraAzim.ToString(), ",", CameraElev.ToString());
            str = string.Concat(str, ",", ValidTarget.ToString(), ",", ImageSensorType.ToString());
            for (int i = 0; i < 4; i++)
                str = string.Concat(str, ",", ImageGroundCornerPostions[i].ToCSV_String());

            str = string.Concat(str, ",", targetInformation[0].ToString());
            str = string.Concat(str, ",", targetInformation[0].Azimuth.ToString());
            str = string.Concat(str, ",", targetInformation[0].Elevation.ToString());
            str = string.Concat(str, ",", targetInformation[0].TargetXYPos.ToCSV_String());
            str = string.Concat(str, ",", targetInformation[0].TargetLatLonPos.ToCSV_String());

            for (int i = 0; i < 4; i++)
                str = string.Concat(str, ",0"); // JAY: Not included for JHART project: TargetCornerPostions[i].ToCSV_String());

            str = string.Concat(str, ",", targetInformation[0].ImgTargetLoc.ToCSV_String());
            for (int i = 0; i < 4; i++)
                str = string.Concat(str, ",0"); // JAY: Not included for JHART project: ImgCorners[i].ToCSV_String());

            str = string.Concat(str, ",", targetInformation[0].TargetHeatGradient.ToString());
            str = string.Concat(str, "\n"); //Terminate in newline
            return str;
        }

        private string ToCSV_String_v1_1()
        {
            xyCoord_t xyPos = new xyCoord_t();
            latLon_t latLonPos = new latLon_t(Latitude, Longitude);
            if (UAVCoordinateSystem.getUAVCoordinateSystem().IsCoordinateSystemValid)
            {
                xyPos = UAVCoordinateSystem.getUAVCoordinateSystem().latLonToXY(latLonPos);
            }
            string str = string.Concat(ImageNumber.ToString(), ",", ImageSize.ToString(), ",", TimeSec.ToString());
            str = string.Concat(str, ",", (DegsPerRad * Latitude).ToString(), ",", (DegsPerRad * Longitude).ToString(), 
                                     ",", xyPos.x.ToString(), ",", xyPos.y.ToString(), ",", AltitudeMSL.ToString(), ",", HeightAGL.ToString());
            str = string.Concat(str, ",", AirSpeed.ToString(), ",", CompassHeading.ToString(), ",", VelNorth.ToString(), ",", VelEast.ToString(), ",", VelDown.ToString());
            str = string.Concat(str, ",", (DegsPerRad * Roll).ToString(), ",", (DegsPerRad * Pitch).ToString(), ",", (DegsPerRad * Yaw).ToString(), 
                                     ",", (DegsPerRad * CameraAzim).ToString(), ",", (DegsPerRad * CameraElev).ToString());
            str = string.Concat(str, ",", ValidTarget.ToString(), ",", ImageSensorType.ToString(), ",", NumTargets.ToString());
            for (int i = 0; i < 4; i++)
                str = string.Concat(str, ",", ImageGroundCornerPostions[i].ToCSV_String());


            // JAY: targetInformation.Length will be the fixed maximum number of targets. (e.g. 4 for the JHART project)
            // The unused targets are cleared so they will just print 0.
            for (int indTarget = 0; indTarget < targetInformation.Length; indTarget++)
            {
                str = string.Concat(str, ",", targetInformation[indTarget].TargetHostileEtcCode.ToString());
                str = string.Concat(str, ",0"); //, targetInformation[indTarget].TargetType.ToString());
                str = string.Concat(str, ",", (DegsPerRad * targetInformation[indTarget].Azimuth).ToString());
                str = string.Concat(str, ",", (DegsPerRad * targetInformation[indTarget].Elevation).ToString());
                str = string.Concat(str, ",", targetInformation[indTarget].TargetXYPos.ToCSV_String());
                str = string.Concat(str, ",", (DegsPerRad * targetInformation[indTarget].TargetLatLonPos.lat).ToString());
                str = string.Concat(str, ",", (DegsPerRad * targetInformation[indTarget].TargetLatLonPos.lon).ToString());
                str = string.Concat(str, ",", targetInformation[indTarget].ImgTargetLoc.ToCSV_String());
            }
            str = string.Concat(str, "\n"); //Terminate in newline
            return str;
        }
    }


    /// <summary>
    /// Image Header Structure
    /// ImageHeaderSyncString: string, 32 bytes, this can be used to find the start of a new image header.
    /// </summary>
    public class ImageHeaderReader
    {
        public int ImageHeaderSize = 512;
        public const string _imageHdrSyncString = "####:!:ImageHeaderSync:!:####...";
        public string ImageHdrSyncString;
        private ByteArrayReader br;
        public ImageHeader ImageHdr;
        private byte[] hdrData;
        public double missionStartTimeSec = 0.0;

        public ImageHeaderReader(int hdrSize)
        {
            ImageHeaderSize = hdrSize;
            hdrData = new byte[hdrSize];
            ImageHdr = new ImageHeader();
            br = new ByteArrayReader(hdrData, EndianOrder_e.big);
        }

        public void Clear()
        {
            br.Idx = 0;
        }

        private bool readHeaderToBuffer(FileStream fileStr)
        {
            bool error = false;
            br.Idx = 0;
            try
            {
                fileStr.Read(hdrData, 0, ImageHeaderSize);
            }
            catch (Exception e)
            {
                MessageBox.Show("Error reading image header, Ex: " + e.Message);
                error = true;
            }
            return error;
        }

        private bool readSyncString()
        {
            ImageHdrSyncString = br.readString(32);
            return string.Compare(ImageHdrSyncString,_imageHdrSyncString) == 0 ? true : false;
        }

        private void readHeader_v1()
        {
            //Clear parameters that may not be set.
            ImageHdr.clearTgtInvalidParams();

            ImageHdr.ImageNumber = br.readInt32();
            ImageHdr.ImageSize = br.readInt32();
            ImageHdr.TimeSec = br.readDouble();
            long MSTicks = (long)(ImageHdr.TimeSec / 100.0e-9);
            long ticksRef = DateTime.Parse("1970-01-01T00:00:00").Ticks;
            MSTicks += ticksRef;
            DateTime msTime = new DateTime(MSTicks);
            ImageHdr.DateTimeStamp = msTime;

            ImageHdr.Latitude = br.readDouble();
            ImageHdr.Longitude = br.readDouble();
            ImageHdr.AltitudeMSL = br.readDouble();
            ImageHdr.HeightAGL = br.readDouble();
            ImageHdr.AirSpeed = br.readDouble();
            ImageHdr.CompassHeading = br.readDouble();
            ImageHdr.VelNorth = br.readDouble();
            ImageHdr.VelEast = br.readDouble();
            ImageHdr.VelDown = br.readDouble();
            ImageHdr.Roll = br.readDouble();
            ImageHdr.Pitch = br.readDouble();
            ImageHdr.Yaw = br.readDouble();
            ImageHdr.CameraAzim = br.readDouble();
            ImageHdr.CameraElev = br.readDouble();

            for (int i = 0; i < 4; i++)
            {
                ImageHdr.ImageGroundCornerPostions[i].x = br.readDouble();
                ImageHdr.ImageGroundCornerPostions[i].y = br.readDouble();
            }

            //Read Target Info if Available.
            ImageHdr.ValidTarget = (int)br.readUInt8();
            if ((SurfaceTarget.ValidationStatus)ImageHdr.ValidTarget != SurfaceTarget.ValidationStatus.NoTarget
                && (SurfaceTarget.ValidationStatus)ImageHdr.ValidTarget != SurfaceTarget.ValidationStatus.InvalidImage)
            {
                ImageHdr.ImageSensorType = (ImageSensorType_e)br.readUInt8();
                ImageHdr.NumTargets = 1;
                //ImageHdr.targetInformation[0].TargetType = (TargetType_e)br.readUInt8();
                br.readUInt8(); // Replaces line above.
                ImageHdr.targetInformation[0].TargetLatLonPos.lat = br.readDouble();
                ImageHdr.targetInformation[0].TargetLatLonPos.lon = br.readDouble();
                ImageHdr.targetInformation[0].TargetXYPos.x = br.readDouble();
                ImageHdr.targetInformation[0].TargetXYPos.y = br.readDouble();
                ImageHdr.targetInformation[0].Azimuth = br.readDouble();
                ImageHdr.targetInformation[0].Elevation = br.readDouble();

                ImageHdr.targetInformation[0].ImgTargetLoc.x = br.readDouble();
                ImageHdr.targetInformation[0].ImgTargetLoc.y = br.readDouble();

                if (ImageHdr.ImageSensorType == ImageSensorType_e.IR)
                {
                    ImageHdr.targetInformation[0].TargetHeatGradient = br.readDouble();
                }
                else
                {
                    ImageHdr.targetInformation[0].TargetColorValue = br.readDouble();
                }
            }
        }

        /// <summary>
        /// Read version 1.1
        /// 
        /// The only difference between versions 1.1 and 1.2 is that the ImageProcTargetStatus has a different
        /// intepretation.
        /// </summary>
        private void readHeader_v1_1()
        {
            //Clear parameters that may not be set.
            ImageHdr.clearTgtInvalidParams();

            ImageHdr.ImageNumber = br.readInt32();
            ImageHdr.ImageSize = br.readInt32();
            ImageHdr.TimeSec = br.readDouble();
            long MSTicks = (long)(ImageHdr.TimeSec / 100.0e-9);
            long ticksRef = DateTime.Parse("1970-01-01T00:00:00").Ticks;
            MSTicks += ticksRef;
            DateTime msTime = new DateTime(MSTicks);
            ImageHdr.DateTimeStamp = msTime;

            ImageHdr.Latitude = br.readDouble();
            ImageHdr.Longitude = br.readDouble();
            ImageHdr.AltitudeMSL = br.readDouble();
            ImageHdr.HeightAGL = br.readDouble();
            ImageHdr.AirSpeed = br.readDouble();
            ImageHdr.CompassHeading = br.readDouble();
            ImageHdr.VelNorth = br.readDouble();
            ImageHdr.VelEast = br.readDouble();
            ImageHdr.VelDown = br.readDouble();
            ImageHdr.Roll = br.readDouble();
            ImageHdr.Pitch = br.readDouble();
            ImageHdr.Yaw = br.readDouble();
            ImageHdr.CameraAzim = br.readDouble();
            ImageHdr.CameraElev = br.readDouble();

            for (int i = 0; i < 4; i++)
            {
                ImageHdr.ImageGroundCornerPostions[i].x = br.readDouble();
                ImageHdr.ImageGroundCornerPostions[i].y = br.readDouble();
            }

            //Read Target Info if Available.
            ImageHdr.ValidTarget = (int)br.readUInt8(); // type is TargetStatus_e in the place this is written.
            ImageHdr.ImageSensorType = (ImageSensorType_e)br.readUInt8();
            ImageHdr.NumTargets = (int)br.readUInt8();
                
            if ((SurfaceTarget.ValidationStatus)ImageHdr.ValidTarget != SurfaceTarget.ValidationStatus.NoTarget
                && (SurfaceTarget.ValidationStatus)ImageHdr.ValidTarget != SurfaceTarget.ValidationStatus.InvalidImage)
            {
                for (int indTarget = 0; indTarget < ImageHdr.NumTargets && indTarget < HOPS_GlobalConstants.MaxNoTargetsPerImage; indTarget++)
                {
                    ImageHdr.targetInformation[indTarget].TargetHostileEtcCode = (ImageProcTargetCode_e)br.readUInt8();
                    ImageHdr.targetInformation[indTarget].TargetLatLonPos.lat = br.readDouble();
                    ImageHdr.targetInformation[indTarget].TargetLatLonPos.lon = br.readDouble();
                    ImageHdr.targetInformation[indTarget].TargetXYPos.x = br.readDouble();
                    ImageHdr.targetInformation[indTarget].TargetXYPos.y = br.readDouble();
                    ImageHdr.targetInformation[indTarget].Azimuth = br.readDouble();
                    ImageHdr.targetInformation[indTarget].Elevation = br.readDouble();
                    ImageHdr.targetInformation[indTarget].ImgTargetLoc.x = br.readDouble();
                    ImageHdr.targetInformation[indTarget].ImgTargetLoc.y = br.readDouble();
                }
            }
        }

        /// <summary>
        /// Read version 1.2.
        /// 
        /// The only difference between versions 1.1 and 1.2 is that the ImageProcTargetStatus has a different
        /// intepretation.
        /// </summary>
        private void readHeader_v1_2()
        {
            //Clear parameters that may not be set.
            ImageHdr.clearTgtInvalidParams();

            ImageHdr.ImageNumber = br.readInt32();
            ImageHdr.ImageSize = br.readInt32();
            ImageHdr.TimeSec = br.readDouble();
            long MSTicks = (long)(ImageHdr.TimeSec / 100.0e-9);
            long ticksRef = DateTime.Parse("1970-01-01T00:00:00").Ticks;
            MSTicks += ticksRef;
            DateTime msTime = new DateTime(MSTicks);
            ImageHdr.DateTimeStamp = msTime;

            ImageHdr.Latitude = br.readDouble();
            ImageHdr.Longitude = br.readDouble();
            ImageHdr.AltitudeMSL = br.readDouble();
            ImageHdr.HeightAGL = br.readDouble();
            ImageHdr.AirSpeed = br.readDouble();
            ImageHdr.CompassHeading = br.readDouble();
            ImageHdr.VelNorth = br.readDouble();
            ImageHdr.VelEast = br.readDouble();
            ImageHdr.VelDown = br.readDouble();
            ImageHdr.Roll = br.readDouble();
            ImageHdr.Pitch = br.readDouble();
            ImageHdr.Yaw = br.readDouble();
            ImageHdr.CameraAzim = br.readDouble();
            ImageHdr.CameraElev = br.readDouble();

            for (int i = 0; i < 4; i++)
            {
                ImageHdr.ImageGroundCornerPostions[i].x = br.readDouble();
                ImageHdr.ImageGroundCornerPostions[i].y = br.readDouble();
            }

            //Read Target Info if Available.
            ImageHdr.ValidTarget = (int)br.readUInt8(); // type is TargetStatus_e in the place this is written.
            ImageHdr.ImageSensorType = (ImageSensorType_e)br.readUInt8();
            ImageHdr.NumTargets = (int)br.readUInt8();

            int indTarget;
            for (indTarget = 0; indTarget < ImageHdr.NumTargets && indTarget < HOPS_GlobalConstants.MaxNoTargetsPerImage; indTarget++)
            {
                ImageHdr.targetInformation[indTarget].TargetHostileEtcCode = (ImageProcTargetCode_e)br.readUInt8();
                ImageHdr.targetInformation[indTarget].TargetLatLonPos.lat = br.readDouble();
                ImageHdr.targetInformation[indTarget].TargetLatLonPos.lon = br.readDouble();
                ImageHdr.targetInformation[indTarget].TargetXYPos.x = br.readDouble();
                ImageHdr.targetInformation[indTarget].TargetXYPos.y = br.readDouble();
                ImageHdr.targetInformation[indTarget].Azimuth = br.readDouble();
                ImageHdr.targetInformation[indTarget].Elevation = br.readDouble();
                ImageHdr.targetInformation[indTarget].ImgTargetLoc.x = br.readDouble();
                ImageHdr.targetInformation[indTarget].ImgTargetLoc.y = br.readDouble();
            }
            for (; indTarget < HOPS_GlobalConstants.MaxNoTargetsPerImage; indTarget++)
            {
                ImageHdr.targetInformation[indTarget].TargetHostileEtcCode = ImageProcTargetCode_e.NoTarget;
                ImageHdr.targetInformation[indTarget].TargetLatLonPos.lat = 0;
                ImageHdr.targetInformation[indTarget].TargetLatLonPos.lon = 0;
                ImageHdr.targetInformation[indTarget].TargetXYPos.x = 0;
                ImageHdr.targetInformation[indTarget].TargetXYPos.y = 0;
                ImageHdr.targetInformation[indTarget].Azimuth = 0;
                ImageHdr.targetInformation[indTarget].Elevation = 0;
                ImageHdr.targetInformation[indTarget].ImgTargetLoc.x = 0;
                ImageHdr.targetInformation[indTarget].ImgTargetLoc.y = 0;
            }
        }


        public bool readImageHeader(FileStream fileStr, string version)
        {
            long hdrStartPos = fileStr.Position;
            if( readHeaderToBuffer(fileStr) )
                return true;

            if (!readSyncString())
            {
                MessageBox.Show("Error reading image sync string. File offset: " + hdrStartPos.ToString());
                return true;
            }

            if(ImageFileHeaderReader.ImageFileVersion_v1.Equals(version))
            {
                readHeader_v1();
            }
            else if (ImageFileHeaderReader.ImageFileVersion_v1_1.Equals(version))
            {
                readHeader_v1_1();
            }
            else if (ImageFileHeaderReader.ImageFileVersion_v1_2.Equals(version))
            {
                readHeader_v1_2();
            }
            else
            {
                MessageBox.Show("Error reading file: Unsuported version: " + version);
                return true; // error
            }
            //Assume ok and return;
            return false; // no error
        }


        public bool writeToFile(string filename)
        {
            bool error = false;
            XmlSerializer xmlSer = null;
            FileStream fs = null;

            try
            {
                fs = File.Open(filename, FileMode.Create, FileAccess.Write, FileShare.None);
                xmlSer = new XmlSerializer(ImageHdr.GetType());
                xmlSer.Serialize(fs, ImageHdr);
                fs.Flush();
                fs.Close();
            }
            catch (Exception e)
            {
                MessageBox.Show("Error storing image header: " + filename + " Ex: " + e.Message);
                error = true;
            }
            return error;
        }

    }

    public class ImageExtractor
    {
        private const string DirSeparator = "\\";
        private int _imageNumber;
        private FileStream _imageFile;
        private FileStream _HopsCvsMetaData = null;
        public FileStream HopsCvsMetaData
        {
            get { return _HopsCvsMetaData; }
            set { _HopsCvsMetaData = value; }
        }
        private string _HopsCvsMetaDataFilename = "HopsImageMetadata.csv";
        public string HopsCvsMetaDataFilename
        {
            get { return _HopsCvsMetaDataFilename; }
            set { _HopsCvsMetaDataFilename = value; }
        }


        private ImageFileHeaderReader _imageFileHeader;
        private ImageHeaderReader _imageHeader;
        private string _storageDirectory;
        private string _imageDirectory;
        private string _imageFilename;
        private string _imageFileBaseName;

        public string StorageDirectory
        {
            get { return _storageDirectory; }
            set { _storageDirectory = value; }
        }

        public string ImageFilename
        {
            get { return _imageFilename; }
            set 
            { 
                _imageFilename = value;
                //Extract base filename
                if (value != null)
                {
                    int n = _imageFilename.LastIndexOf(DirSeparator);
                    if (n > 0)
                    {
                        _imageDirectory = _imageFilename.Substring(0, n);
                    }
                    else
                    {
                        _imageDirectory = "";
                    }
                    ++n;
                    int m = _imageFilename.Length - n;
                    if (n > 1 && m > 0)
                    {
                        _imageFileBaseName = _imageFilename.Substring(n, m);
                    }
                    else
                    {
                        _imageFileBaseName = _imageFilename;
                    }

                    n = _imageFileBaseName.LastIndexOf(".");
                    if (n > 0)
                    {
                        _imageFileBaseName = _imageFileBaseName.Substring(0, n);
                    }
                }
            }
        }

        public ImageExtractor()
        {
            _imageNumber = 0;
            _imageFile = null;
            _imageFileHeader = new ImageFileHeaderReader();
            _imageHeader = null;
            _imageFilename = null;
            _storageDirectory = null;
        }

        public ImageExtractor(string imagefilename, string extractDir)
        {
            _imageNumber = 0;
            _imageFile = null;
            _imageFileHeader = new ImageFileHeaderReader();
            _imageHeader = null;
            ImageFilename = imagefilename;
            StorageDirectory = extractDir;
        }

        public bool createDirectory(string dirName)
        {
            bool error = false;
            try
            {
                DirectoryInfo dinfo = new DirectoryInfo(dirName);
                if (!dinfo.Exists)
                {
                    dinfo.Create();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Could not create directory: " + dirName + " Ex:" + ex.Message);
                error = true;
            }
            return error;
        }

        public bool openNewImageFile(string filename)
        {
            bool error = false;
            if (_imageFile != null)
            {
                _imageFile.Close();
                _imageFile = null;
            }
            try
            {
                _imageFile = new FileStream(filename, FileMode.Open, FileAccess.Read);
                ImageFilename = filename;
            }
            catch (Exception ioException)
            {
                MessageBox.Show("Cannot open image file: " + filename
                        + "Exception: " + ioException.Message);
                error = true;
            }
            return error;
        }

        public void closeImageFile()
        {
            if (_imageFile != null)
            {
                _imageFile.Close();
                _imageFile = null;
            }
        }

        public bool extractFileHeader()
        {
            bool error = false;
            string hdrFilename = "";
            if (_imageFile == null)
            {
                MessageBox.Show("Error: Image File is not open.");
                return true;
            }
            try
            {
                error = _imageFileHeader.readFileHeader(_imageFile);
                //Write File header info to a file.
                if (_storageDirectory != null)
                {
                    hdrFilename = string.Concat(_storageDirectory, DirSeparator);
                }
                else
                {
                    hdrFilename = string.Concat(_imageDirectory, DirSeparator);
                }
                hdrFilename = string.Concat(hdrFilename, _imageFileBaseName, "_fileHdr.xml");
                _imageFileHeader.writeToFile(hdrFilename);

                //Setup the UAV Coordinate system base on the Center Lat/Lon info
                UAVCoordinateSystem.getUAVCoordinateSystem().setupCoordinateSystemPostProcessing(_imageFileHeader.FileHeader.CoordSysCenter, 
                                                                                                _imageFileHeader.FileHeader.LatLonToXYScaleFactors);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error extracting the Image File Header, Exception: " + ex.Message);
                error = true;
            }
            return error;
        }

        public bool extractImageHeader(FileStream csvMetaDataFile)
        {
            bool error = false;
            string ImageHdrFilename = "";
            if (_imageFile == null)
            {
                MessageBox.Show("Error: Image File is not open.");
                return true;
            }
            if (_imageHeader == null)
            {
                _imageHeader = new ImageHeaderReader(_imageFileHeader.ImageHeaderSize);
            }
            try
            {
                error = _imageHeader.readImageHeader(_imageFile,_imageFileHeader.ImageFileVersion);
                //Write File header info to a file.
                if (!error)
                {
                    if (csvMetaDataFile != null)
                    {
                        ASCIIEncoding ascii = new ASCIIEncoding();
                        byte[] asciiBytes = ascii.GetBytes(_imageHeader.ImageHdr.ToCSV_String(_imageFileHeader.ImageFileVersion));
                        _HopsCvsMetaData.Write(asciiBytes, 0, asciiBytes.Length);
                    }
                    else
                    {
                        _imageNumber = _imageHeader.ImageHdr.ImageNumber;
                        if (_storageDirectory != null)
                        {
                            ImageHdrFilename = string.Concat(_storageDirectory, DirSeparator);
                        }
                        else
                        {
                            ImageHdrFilename = string.Concat(_imageDirectory, DirSeparator);
                        }
                        ImageHdrFilename = string.Concat(ImageHdrFilename, _imageFileBaseName, "_HdrNo_", _imageNumber.ToString("0000"), ".xml");
                        _imageHeader.writeToFile(ImageHdrFilename);
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error extracting the Image Header, Exception: " + ex.Message);
                error = true;
            }
            return error;
        }

        public bool extractImageFile()
        {
            bool error = false;
            string imageFilename = "";
            if (_imageFile == null)
            {
                MessageBox.Show("Error: Image File is not open.");
                return true;
            }
            if (_imageHeader == null)
            {
                return true;       //Error we need the image file info for image size.
            }
            int imageSize = _imageHeader.ImageHdr.ImageSize;
            int imageNo = _imageHeader.ImageHdr.ImageNumber;
            if( imageSize < 100 )
            {
                MessageBox.Show("Error: Invalid Image Size: " + imageSize.ToString());
                return true;
            }
            try
            {
                byte[] image = new byte[imageSize];
                _imageFile.Read(image, 0, imageSize);

                //Write File header info to a file.
                if (_storageDirectory != null)
                {
                    imageFilename = string.Concat(_storageDirectory, DirSeparator);
                }
                else
                {
                    imageFilename = string.Concat(_imageDirectory, DirSeparator);
                }
                imageFilename = string.Concat(imageFilename, _imageFileBaseName, "_ImgNo_",
                                            imageNo.ToString("0000"), ".", 
                                            _imageFileHeader.FileHeader.ImageFileExt);

                FileStream fs = File.Open(imageFilename, FileMode.Create, FileAccess.Write, FileShare.None);
                fs.Write(image, 0, imageSize);
                fs.Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error extracting or writing the Image No: " + imageNo.ToString() 
                                + " Exception: " + ex.Message);
                error = true;
            }
            return error;
        }


        public int extractImages()
        {
            int NoImagesExtracted = 0;
            if (_imageFile == null )
            {
                MessageBox.Show("Open the Image File before Extracting Images");
                return NoImagesExtracted;
            }
            if (_storageDirectory != null && _storageDirectory != "" )
            {
                if (createDirectory(_storageDirectory))
                    return NoImagesExtracted;
            }
            else
            {
                _storageDirectory = null;  //files will be stored in ImageFile Dir.
            }
            if (extractFileHeader())
            {
                return NoImagesExtracted;
            }

            //Go to the start of the first Image
            int imgOffset = _imageFileHeader.FileHeaderSize;
            int imgHdrSize = _imageFileHeader.ImageHeaderSize;
            _imageFile.Seek(imgOffset, SeekOrigin.Begin);

            _imageHeader = new ImageHeaderReader(imgHdrSize);
            
            string csvMetadataFilename;
            if (_storageDirectory != null)
            {
                csvMetadataFilename = string.Concat(_storageDirectory, DirSeparator, _HopsCvsMetaDataFilename);
            }
            else
            {
                csvMetadataFilename = string.Concat(_imageDirectory, DirSeparator, _HopsCvsMetaDataFilename);

            }

            try
            {
                if (File.Exists(csvMetadataFilename))
                {
                    _HopsCvsMetaData = new FileStream(csvMetadataFilename, FileMode.Append, FileAccess.Write);
                }
                else
                {
                    _HopsCvsMetaData = new FileStream(csvMetadataFilename, FileMode.CreateNew, FileAccess.Write);

                    ASCIIEncoding ascii = new ASCIIEncoding();
                    byte[] asciiBytes = ascii.GetBytes(ImageHeader.CSVHdrString(_imageFileHeader.ImageFileVersion));
                    _HopsCvsMetaData.Write(asciiBytes, 0, asciiBytes.Length);
                }
            }
            catch(Exception ex)
            {
                MessageBox.Show("Error creating/opening file: " + csvMetadataFilename
                    + " Ex: " + ex.Message);
                return 0;
            }

            bool endOfFile = false;
            while (!endOfFile)
            {
                if (extractImageHeader(_HopsCvsMetaData))
                {
                    break;      //Error reading header... stop here.
                }
                if (extractImageFile())
                {
                    break;
                }
                ++NoImagesExtracted;
                if (_imageFile.Position + 100 > _imageFile.Length)
                {
                    endOfFile = true;
                }
            }
            if (_HopsCvsMetaData != null)
            {
                _HopsCvsMetaData.Flush();
                _HopsCvsMetaData.Close();
            }
            return NoImagesExtracted;
        }



    }
}
