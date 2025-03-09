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

#include "ImagePlusMetadataFileHeaders.h"
#include <memory.h>

using namespace VidereUtils;
using namespace std;

namespace VidereImageprocessing
{

    const std::string ImagePlusMetadataFileHeader::ImageFileTitle = "[Falcon Image Plus Metadata]";

    void ImagePlusMetadataFileHeader::ClearByteArray()
    {
        memset(_byteArray, 0, IMAGEPLUSMETADATAFILEHEADERSIZE);
    }

    char* ImagePlusMetadataFileHeader::CreateFileHeader()
    {
        VidereUtils::ByteArrayWriterVidere bw(_byteArray, IMAGEPLUSMETADATAFILEHEADERSIZE, Rabit::EndianOrder_e::Endian_Big);
        ComputerEndianess = GetSystemEndianness();

        GeoCoordinateSystem *gcs = GeoCoordinateSystem::GetGeoCoordinateSystemReference();
        GeoCoordinateSystemConversionType = gcs->GetConversionType();
        GeoReferenceLocation = gcs->ReferenceLatLonAltLocation();
        LatitudeRadToYCF = gcs->GetLatitudeRadToYConversionFactor();
        LongitudeRadToXCF = gcs->GetLongitudeRadToXConversionFactor();

        bw.writeString(ImageFileTitle);
        bw.writeUInt32(FileVersionNumber);
        bw.writeUInt32(FileHeaderSize);
        bw.writeUInt32(ImageHeaderSize);
        bw.writeUInt32((uint32_t)ComputerEndianess);
        bw.writeUInt32((uint32_t)SensorType);
        bw.writeUInt32((uint32_t)GeoCoordinateSystemConversionType);
        bw.writeLatLonAlt(GeoReferenceLocation);
        bw.writeDouble(LatitudeRadToYCF);
        bw.writeDouble(LongitudeRadToXCF);

        MetadataSize = bw.Idx;
        return (char*)_byteArray;
    }

    bool ImagePlusMetadataFileHeader::ReadFileHeader()
    {
        bool error = false;
        VidereUtils::ByteArrayReaderVidere br(_byteArray, IMAGEPLUSMETADATAFILEHEADERSIZE, Rabit::EndianOrder_e::Endian_Big);

        br.readString(&Title);
        //Check the Image header title... error if not equal to
        //the set title.
        error = Title != ImageFileTitle;

        FileVersionNumber = br.readUInt32();
        FileHeaderSize = br.readUInt32();
        ImageHeaderSize = br.readUInt32();
        ComputerEndianess = (EndianOrder_e)br.readUInt32();
        SensorType = (SensorType_e)br.readUInt32();
        GeoCoordinateSystemConversionType = (GeoCoordinateSystemConversionType_e)br.readUInt32();
        GeoReferenceLocation = br.readLatLonAlt();
        LatitudeRadToYCF = br.readDouble();
        LongitudeRadToXCF = br.readDouble();

        MetadataSize = br.Idx;
        return error;
    }

    bool ImagePlusMetadataFileHeader::SetupGeoCoordinateSystemFromGeoRefInfo()
    {
        GeoCoordinateSystem *gcs = GeoCoordinateSystem::GetGeoCoordinateSystemReference();
        if(GeoCoordinateSystemConversionType == GeoCoordinateSystemConversionType_e::Linear)
        {
            return gcs->SetupLinearGeoCoordinateSystemFromConvFactors(GeoReferenceLocation,
                                                                 LatitudeRadToYCF, LongitudeRadToXCF);
        }
        else
        {
            return gcs->SetupGeoCoordinateSystem(GeoReferenceLocation,
                                                 GeoCoordinateSystemConversionType);
        }
    }


    /**********  Image Header ***************************/
    const std::string ImagePlusMetadataHeader::ImageHeaderTitle = "[::ImageDataStart::]";

    void ImagePlusMetadataHeader::ClearByteArray()
    {
        memset(_byteArray, 0, IMAGEPLUSMETADATAFILEHEADERSIZE);
    }

    uint32_t ImagePlusMetadataHeader::SetCVMatParamsComputeMatSize(ImagePlusMetadataMessage &imgMdMsg)
    {
        uint32_t size = 0;
        cvMatCols = (uint16_t)imgMdMsg.ImageFrame.cols;
        cvMatRows = (uint16_t)imgMdMsg.ImageFrame.rows;
        cvMatElementType = (uint16_t)imgMdMsg.ImageFrame.type();
        cvMatElementSize = (uint16_t)imgMdMsg.ImageFrame.elemSize();
        cvMatNumChannels = (uint16_t)imgMdMsg.ImageFrame.channels();
        size = cvMatCols * cvMatRows * cvMatElementSize;
        return size;
    }


    char* ImagePlusMetadataHeader::CreateImageHeader(ImagePlusMetadataMessage &imgMdMsg)
    {
        VidereUtils::ByteArrayWriterVidere bw(_byteArray, IMAGEPLUSMETADATAIMAGEHEADERSIZE, Rabit::EndianOrder_e::Endian_Big);

       bw.writeString(ImageHeaderTitle);
        ImageNumber = imgMdMsg.ImageNumber;
        bw.writeUInt32((uint32_t)imgMdMsg.ImageNumber);
        bw.writeUInt16((uint16_t)ImageStorageType);
        bw.writeUInt16((uint16_t)ImageFormatType);
        bw.writeUInt32((uint32_t)ImageSize);
        bw.writeUInt16((uint16_t)imgMdMsg.ImageNoPixelsWide);
        bw.writeUInt16((uint16_t)imgMdMsg.ImageNoPixelsHigh);

        //Image Corner Locations (4*3*8 = 96)
        for(int i = 0; i < 4; i++)
            bw.writeXYZ(imgMdMsg.ImageCorners[i]);

        //Write the OpenCV Matrix Items out so it can be reconstructed if
        //needed.
        SetCVMatParamsComputeMatSize(imgMdMsg);

        bw.writeUInt16(cvMatCols);
        bw.writeUInt16(cvMatRows);
        bw.writeUInt16(cvMatElementType);
        bw.writeUInt16(cvMatElementSize);
        bw.writeUInt16(cvMatNumChannels);

        imgMdMsg.VehicleInertialStates.SerializeToByteArray(bw); //112 Bytes
        imgMdMsg.CameraOrientation.SerializeToByteArray(bw);  //49 Bytes

        MetadataSize = bw.Idx;
        return (char*)_byteArray;
    }

    bool ImagePlusMetadataHeader::ReadImageHeader(ImagePlusMetadataMessage *imgMdMsg)
    {
        bool error = false;
        VidereUtils::ByteArrayReaderVidere br(_byteArray, IMAGEPLUSMETADATAIMAGEHEADERSIZE, Rabit::EndianOrder_e::Endian_Big);

        br.readString(&ImgTitle);
        //Check the Image header title... error if not equal to
        //the set title.
        error = ImgTitle != ImageHeaderTitle;

        ImageNumber = br.readInt32();
        imgMdMsg->ImageNumber = ImageNumber;
        ImageStorageType = (ImageStorageType_e)br.readInt16();
        ImageFormatType = (ImageFormatType_e)br.readInt16();

        ImageSize = br.readInt32();
        imgMdMsg->ImageNoPixelsWide = br.readInt16();
        imgMdMsg->ImageNoPixelsHigh = br.readInt16();

        for(int i = 0; i < 4; i++)
            imgMdMsg->ImageCorners[i] = br.readXYZ();

        cvMatCols = br.readUInt16();
        cvMatRows = br.readUInt16();
        cvMatElementType = br.readUInt16();
        cvMatElementSize = br.readUInt16();
        cvMatNumChannels = br.readUInt16();

        imgMdMsg->VehicleInertialStates.ReadFromByteArray(br);
        imgMdMsg->CameraOrientation.ReadFromByteArray(br);

        MetadataSize = br.Idx;
        return error;
    }

}