/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Oct 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#ifndef VIDERE_DEV_IMAGEPLUSMETADATAFILEHEADERS_H
#define VIDERE_DEV_IMAGEPLUSMETADATAFILEHEADERS_H

#include "global_defines.h"
#include <opencv2/core.hpp>
#include "LatLonAltStruct.h"
#include "XYZCoord_t.h"
#include "RollPitchYaw_t.h"
#include "AzimuthElevation_t.h"
#include "config_data.h"
#include "../../Utils/logger.h"
#include "ByteArrayReaderWriterVidere.h"
#include "GeoCoordinateSystem.h"
#include "image_plus_metadata_message.h"

using namespace videre;
using namespace GeoCoordinateSystemNS;
using namespace MathLibsNS;

#define IMAGEPLUSMETADATAFILEVERSIONNUMBER 1

//The header sizes are larger than required... but this makes
//it easer to add or expand metadata without changing the
//structure of the reader.
#define IMAGEPLUSMETADATAFILEHEADERSIZE 512
#define IMAGEPLUSMETADATAIMAGEHEADERSIZE 512


namespace VidereImageprocessing
{


    ///Each Image Plus Metadata file has a header to define
    ///various parameters and characteristics of the file.
    ///This class defines those parameters and will serialize and
    ///deserialize the header.
    class ImagePlusMetadataFileHeader
    {
    public:
        static const std::string ImageFileTitle;

        std::string Title;
        //The File Version can be used to help in the reading of the
        //file.  Different version of the file may have different information
        //in the metadata.
        uint32_t FileVersionNumber = IMAGEPLUSMETADATAFILEVERSIONNUMBER;

        //The Size of the file's Header.  This should remain fixed even if
        //the version of the file changes.  The header size was choosen
        //large enough to handle most changes.  If more information is required
        //if would be a good idea to add all new information starting at
        //this boundary and above... so that the reader can alway start by
        //reading the base header file size.
        uint32_t FileHeaderSize = IMAGEPLUSMETADATAFILEHEADERSIZE;

        //The size of the metadata assocated with each image.  This is expected
        //to change in size with different file versions.
        uint32_t ImageHeaderSize = IMAGEPLUSMETADATAIMAGEHEADERSIZE;

        //The Endianess of the Computer generating the Image Plus Metadata
        //File.  This information is required because the Metadata information
        //is stored directly as a blob, and the Endianess of the data will
        //depend on the computer operating system.
        Rabit::EndianOrder_e ComputerEndianess;

        //Image Sensor Type:
        SensorType_e SensorType = SensorType_e::EO;

        //Geo-reference Lat/Lon Center.  This is the center point used for
        //the geo-reference system.  Currently this does not change over the
        //course of the mission and can be used to initialized a Geo-reference system
        //when post analyzing the data.
        GeoCoordinateSystemConversionType_e GeoCoordinateSystemConversionType;
        LatLonAltCoord_t GeoReferenceLocation;

        //If a Linear GeoCoordinate System is used... the following
        //conversion factors will be valid.
        double LatitudeRadToYCF = 0.0;
        double LongitudeRadToXCF = 0.0;

        //Number of bytes read/written to the header.
        int MetadataSize = 0;

    private:
        Rabit::byte _byteArray[IMAGEPLUSMETADATAFILEHEADERSIZE];

    public:
        ImagePlusMetadataFileHeader()
        {}

        void ClearByteArray();

        char* CreateFileHeader();

        char* GetFileHeaderArray()
        {
            return (char*)_byteArray;
        }

        bool ReadFileHeader();

        bool SetupGeoCoordinateSystemFromGeoRefInfo();

    };

    enum ImageStorageType_e
    {
        IST_None,
        IST_JPEG,
        IST_OpenCVMatRGB,
        IST_OpenCVMatLum,
    };

    ///Image Plus Metadata header
    ///Information stored before each image.
    ///An image can be any size... including 0 bytes.
    class ImagePlusMetadataHeader
    {
    public:
        //A fixed string is included at the start of each Metadata section...
        //This would allow finding the start of a new metadata plus image
        //block even if part of the file was corrupt.
        static const std::string ImageHeaderTitle;

        std::string ImgTitle;

        const uint32_t ImageHeaderSize = IMAGEPLUSMETADATAIMAGEHEADERSIZE;

        uint32_t ImageNumber = 0;

        //Image Size in Bytes
        uint32_t ImageSize = 0;

        //Image Storage Type
        //An image can be stored in various ways...
        //   1) JPEG byte array
        //   2) OpenCV Matrix (RGB)  (3 channel)
        //   3) OpenCV Matrix (YUV)  (3 channel)
        //   4) OpenCV Matrix (Lumanance)
        ImageStorageType_e ImageStorageType = ImageStorageType_e::IST_JPEG;

        ImageFormatType_e ImageFormatType = ImageFormatType_e::ImgFType_JPEG;

        //OpenCV Matrix Parameters... use for reconstruction
        //of an OpenCV matrix if required.
        uint16_t cvMatCols = 0;
        uint16_t cvMatRows = 0;
        uint16_t cvMatElementType = 0;
        uint16_t cvMatElementSize = 0;
        uint16_t cvMatNumChannels = 0;

        //Number of bytes read/written in the in the Metadata Header;
        int MetadataSize = 0;

    private:
        Rabit::byte _byteArray[IMAGEPLUSMETADATAIMAGEHEADERSIZE];

    public:
        ImagePlusMetadataHeader()
        {}

        void ClearByteArray();

        //Get a pointer to the Image Header Array...
        //this assumes CreateImageHeader has been called.
        char* GetImageHeaderArray()
        {
            return (char*)_byteArray;
        }

        uint32_t SetCVMatParamsComputeMatSize(ImagePlusMetadataMessage &imgMdMsg);

        char* CreateImageHeader(ImagePlusMetadataMessage &imgMdMsg);

        bool ReadImageHeader(ImagePlusMetadataMessage *imgMdMsg);

    };


}


#endif //VIDERE_DEV_IMAGEPLUSMETADATAFILEHEADERS_H
