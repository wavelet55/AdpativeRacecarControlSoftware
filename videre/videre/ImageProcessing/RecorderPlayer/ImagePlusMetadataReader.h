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

#ifndef VIDERE_DEV_IMAGEPLUSMETADATAREADER_H
#define VIDERE_DEV_IMAGEPLUSMETADATAREADER_H

#include "global_defines.h"
#include <opencv2/core.hpp>
#include "LatLonAltStruct.h"
#include "XYZCoord_t.h"
#include "RollPitchYaw_t.h"
#include "AzimuthElevation_t.h"
#include "config_data.h"
#include "../../Utils/logger.h"
#include "CompressedImageMessage.h"
#include "ImagePlusMetadataFileHeaders.h"
#include <boost/filesystem.hpp>

using namespace videre;
using namespace GeoCoordinateSystemNS;
using namespace MathLibsNS;

namespace VidereImageprocessing
{
    enum ImageReturnType_e
    {
        IRT_Error,        //Error reading Image and / or MetaData...
        IRT_EndOfImages,  //End of Images in File(s)
        IRT_MetadataOnly, //No Image info
        IRT_MD_RawImage,  //Raw image returned in the ImagePlusMetadataMessage
        IRT_MD_CompressedImgage
    };

    //Read Image Plus Metadata files
    //An iterator allows Images plus the related metadata to be
    //read one at a time.
    class ImagePlusMetadataReader
    {
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        const int MegaByte = 1024 * 1024;
        const std::string IPM_EXTENSION_STR = "imd";

        bool _imageDirExists = false;
        ImagePlusMetadataFileHeader _imageFileHeader;
        ImagePlusMetadataHeader _imageHeader;
        std::string _imageDirname;
        std::string _baseFilename; /* Base filename*/
        boost::filesystem::path _currentFilePath; /* Full path to current video */
        std::ifstream _imageFile;
        size_t _imageFileSize = 0;

        bool _isImagePlusMetadataFile = false;

        std::vector<boost::filesystem::path> _listOfImageFiles;
        int _imageFileIndex = 0;

        int _NumberOfImagesReadIn = 0;

        bool _EndOfImages = false;

    public:
        //If set to true... will loop back to the start of the images
        //in the first file after reaching the end of all the images.
        bool LoopBackToStartOfImages = false;


    public:
        ImagePlusMetadataReader(std::shared_ptr<ConfigData> config);

        ~ImagePlusMetadataReader();

        int GetNumberOfImagesReadIn()
        {
            return _NumberOfImagesReadIn;
        }

        bool IsEndOfImages()
        {
            return _EndOfImages;
        }

        std::string GetIPMFileExtention()
        {
            return IPM_EXTENSION_STR;
        }

        //Get a Sorted List of Image Plus Metadata files located in the
        //given directory.  If the fileExt is null or empty, the standard
        //default extention is used.
        //Returns the number of files found.
        int GetListOfFilesFromDirectory(const std::string &dir,
                                        const std::string *fileExt = nullptr);

        bool openNextImageFile();

        void closeImageFile();

        void resetToStart()
        {
            closeImageFile();
            _imageFileIndex = 0;
            _NumberOfImagesReadIn = 0;
            _EndOfImages = false;
        }

        //Read the next Image Plus Metadata from the list of files in the directory.
        //The metadata will be loaded into the imagePlusMetadataMsg.
        //An image can be either a compressed image, in which case the compressedImageMsg.
        //If the image is a raw image (OpenCV Mat format), then the image will be
        //loaded into the imagePlusMetadataMsg.ImageFrame.
        //The return enum will provide the details on what was read in.
        //Note:  an ImagePlusMetadataFile can contain a mixture of image types.
        ImageReturnType_e ReadNextImagePlusMetadata(ImagePlusMetadataMessage *imagePlusMetadataMsg,
                                      CompressedImageMessage *compressedImageMsg);

    private:
        //Read the Header at the begining of a ImagePlusMetadata File.
        //Returns true if error, false if header read ok.
        bool ReadFileHeader();

        //Read the Image Header located before each image.
        //Returns true if error, false if header read ok.
        bool ReadImageHeader(ImagePlusMetadataMessage *imagePlusMetadataMsg);


        size_t GetNumberOfBytesRemainingInFile();

        //Returns true if there is not enough room in the file for
        //another metadata section.
        bool CheckEndOfImageFile();

        //Read the compressed Image at the current file location.
        //Return true if error, false otherwise.
        bool ReadCompressedImage(CompressedImageMessage *compressedImageMsg,
                                              int imageSize);

        //Read a raw OpenCV Mat Image from the image file.
        //This assumes the _imageHeader information has already been
        //read in correctly.
        //Note:  Some OpenCV Mat items could be system Endianesss dependant.
        //This should not be an issue for RGB or YUV images where the underlying
        //data is byte oriented.  I could be an issue where the pixel data
        //size is > 1 byte, such as in a Lumance Image.
        //Currently no effort has is taken to fix the endianess issue.. it
        //will only show up if the inmage is read on a differnet operating system
        //(Linux vs Windows) and only if the underlying pixel data is > 1 byte.
        bool ReadRawOpenCVMatImage(cv::Mat *imgMatPtr);

    };
}

#endif //VIDERE_DEV_IMAGEPLUSMETADATAREADER_H
