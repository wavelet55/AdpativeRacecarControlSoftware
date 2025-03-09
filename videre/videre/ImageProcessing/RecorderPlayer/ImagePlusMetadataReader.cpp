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

#include "ImagePlusMetadataReader.h"
#include "FileUtils.h"

using namespace std;


namespace VidereImageprocessing
{

    ImagePlusMetadataReader::ImagePlusMetadataReader(std::shared_ptr<ConfigData> config)
            :_imageFileHeader(),
             _imageHeader(),
             _listOfImageFiles()
    {
        _config_sptr = config;
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        _isImagePlusMetadataFile = false;
        _imageDirExists = false;
        _imageFileIndex = 0;
        _NumberOfImagesReadIn = 0;
        _EndOfImages = false;
        _baseFilename = _config_sptr->GetConfigStringValue("video_recording.base_name", "ImagePlusMetadata");

    }

    size_t ImagePlusMetadataReader::GetNumberOfBytesRemainingInFile()
    {
        size_t count = 0;
        if(_imageFile.is_open())
        {
            count = _imageFileSize - _imageFile.tellg();
            count = count < 0 ? 0 : count;
        }
        return count;
    }

    ImagePlusMetadataReader::~ImagePlusMetadataReader()
    {
        closeImageFile();
    }

    bool ImagePlusMetadataReader::openNextImageFile()
    {
        bool fileOpened = false;
        try
        {
            //First ensure last file is closed
            closeImageFile();

            //Check the file index and ensure it is less
            //than the number of available files
            while(!fileOpened && (_imageFileIndex < _listOfImageFiles.size()))
            {
                _currentFilePath = _listOfImageFiles[_imageFileIndex];
                ios_base::openmode fileMode = ios_base::in | ios_base::binary;
                _imageFile.open(_currentFilePath.c_str(), fileMode);
                //Ensure we are at the beginning of the file
                _imageFile.seekg (0, std::ios::end);
                _imageFileSize = _imageFile.tellg();
                _imageFile.seekg (0, std::ios::beg);
                ++_imageFileIndex;
                fileOpened = true;

                if( ReadFileHeader() )
                {
                    //If the Header cannot be read... close the file
                    //and try the next one.
                    LOGWARN("ImagePlusMetadataReader:openNewImageFile: Could not read File Header in file: " << _currentFilePath );
                    fileOpened = false;
                    closeImageFile();
                }
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("ImagePlusMetadataReader:openNewImageFile: Exception: " << e.what());
            closeImageFile();
            fileOpened = false;
        }
        return fileOpened;
    }


    void ImagePlusMetadataReader::closeImageFile()
    {
        try
        {
            if(_imageFile.is_open())
            {
                _imageFile.close();
                _imageFileSize = 0;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("ImagePlusMetadataReader:closeImageFile: Exception: " << e.what());
        }
    }

    //Get a sorted list of image plus metadata files.
    int ImagePlusMetadataReader::GetListOfFilesFromDirectory(const std::string &dir,
                                                             const std::string *fileExt)
     {
        int numOfFiles = 0;
        _isImagePlusMetadataFile = true;
         _NumberOfImagesReadIn = 0;
         _EndOfImages = false;
         std::string ipmExt = IPM_EXTENSION_STR;
         if( fileExt != nullptr && fileExt->size() > 0)
         {
             ipmExt = *fileExt;
         }
         numOfFiles = VidereFileUtils::GetListFilesInDirectory(&_listOfImageFiles, dir,
                                                               ipmExt, "", true);
        return numOfFiles;
    }

    bool ImagePlusMetadataReader::ReadFileHeader()
    {
        bool error = true;
        if(_imageFile.is_open())
        {
            try
            {
                char *byteArray = _imageFileHeader.GetFileHeaderArray();
                int numBytesToRead = _imageFileHeader.FileHeaderSize;
                if( GetNumberOfBytesRemainingInFile() >= numBytesToRead)
                {
                    _imageFile.read(byteArray, numBytesToRead);
                    error = _imageFileHeader.ReadFileHeader();
                }
                else
                {
                    //This should not normally occur.
                    closeImageFile();
                }
            }
            catch (std::exception &e)
            {
                LOGERROR("ImagePlusMetadataReader:ReadFileHeader: Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }

    bool ImagePlusMetadataReader::ReadImageHeader(ImagePlusMetadataMessage *imagePlusMetadataMsg)
    {
        bool error = true;
        size_t fileOffset = 0;
        if(_imageFile.is_open())
        {
            try
            {
                fileOffset = _imageFile.tellg();

                char *byteArray = _imageHeader.GetImageHeaderArray();
                int numBytesToRead = _imageFileHeader.ImageHeaderSize;
                if (numBytesToRead == 0 || numBytesToRead > _imageHeader.ImageHeaderSize)
                {
                    numBytesToRead = _imageHeader.ImageHeaderSize;
                }

                if (GetNumberOfBytesRemainingInFile() >= numBytesToRead)
                {
                    _imageFile.read(byteArray, numBytesToRead);
                    error = _imageHeader.ReadImageHeader(imagePlusMetadataMsg);
                    if(error)
                    {
                        LOGWARN("ImagePlusMetadataReader: Error reading image header, file: "
                             << _currentFilePath << " Byte Offset: " << fileOffset);
                    }
                }
            }
            catch (std::exception &e)
            {
                LOGERROR("ImagePlusMetadataReader:ReadImageHeader: Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }


    bool ImagePlusMetadataReader::CheckEndOfImageFile()
    {
        bool endOfFile = true;
        if(_imageFile.is_open())
        {
            int numBytesToRead = _imageFileHeader.ImageHeaderSize;
            if (numBytesToRead == 0 || numBytesToRead > _imageHeader.ImageHeaderSize)
            {
                numBytesToRead = _imageHeader.ImageHeaderSize;
            }
            endOfFile = GetNumberOfBytesRemainingInFile() < numBytesToRead;
        }
        return endOfFile;
    }

    ImageReturnType_e ImagePlusMetadataReader::ReadNextImagePlusMetadata(ImagePlusMetadataMessage *imagePlusMetadataMsg,
                                                CompressedImageMessage *compressedImageMsg)
    {
        ImageReturnType_e rtype = ImageReturnType_e::IRT_Error;
        bool metadataRead = false;
        if (CheckEndOfImageFile())
        {
            if (!openNextImageFile())
            {
                if(LoopBackToStartOfImages)
                {
                    //Restart to the beginning of the first file.
                    resetToStart();
                    if (!openNextImageFile())
                    {
                        _EndOfImages = true;
                        return ImageReturnType_e::IRT_EndOfImages;
                    }
                }
                else
                {
                    _EndOfImages = true;
                    return ImageReturnType_e::IRT_EndOfImages;
                }
            }
        }

        if (ReadImageHeader(imagePlusMetadataMsg))
        {
            rtype = ImageReturnType_e::IRT_Error;
            return rtype;
        }

        compressedImageMsg->ImageNumber = imagePlusMetadataMsg->ImageNumber;

        switch(_imageHeader.ImageStorageType)
        {
            case ImageStorageType_e::IST_None:
                rtype = ImageReturnType_e::IRT_MetadataOnly;
                break;

            case ImageStorageType_e::IST_JPEG:
                compressedImageMsg->ImageNumber = _imageHeader.ImageNumber;
                compressedImageMsg->ImageFormatType = _imageHeader.ImageFormatType;
                compressedImageMsg->GpsTimeStampSec = imagePlusMetadataMsg->ImageCaptureTimeStampSec;
                if(ReadCompressedImage(compressedImageMsg, _imageHeader.ImageSize) )
                {
                    rtype = ImageReturnType_e::IRT_Error;
                }
                else
                {
                    rtype = ImageReturnType_e::IRT_MD_CompressedImgage;
                }
                break;

            case ImageStorageType_e::IST_OpenCVMatRGB:
                if( ReadRawOpenCVMatImage(&imagePlusMetadataMsg->ImageFrame) )
                {
                    rtype = ImageReturnType_e::IRT_Error;
                }
                else
                {
                    rtype = ImageReturnType_e::IRT_MD_RawImage;
                }
                break;

            case ImageStorageType_e::IST_OpenCVMatLum:
                if( ReadRawOpenCVMatImage(&imagePlusMetadataMsg->ImageFrame) )
                {
                    rtype = ImageReturnType_e::IRT_Error;
                }
                else
                {
                    rtype = ImageReturnType_e::IRT_MD_RawImage;
                }
                break;

            default:
                LOGWARN("ImagePlusMetadataReader::ReadNextImagePlusMetadata unsupported image type: "
                                << _imageHeader.ImageStorageType);
                rtype = ImageReturnType_e::IRT_Error;
        }

        return rtype;
    }

    bool ImagePlusMetadataReader::ReadCompressedImage(CompressedImageMessage *compressedImageMsg,
                                          int imageSize)
    {
        bool error = true;
        size_t fileOffset1 = 0;
        size_t fileOffset2 = 0;
        int delFileOffset = 0;
        int vecSize = 0;
        if(_imageFile.is_open())
        {
            try
            {
                fileOffset1 = _imageFile.tellg();
                if (GetNumberOfBytesRemainingInFile() >= imageSize)
                {
                    compressedImageMsg->ImageBuffer.resize(imageSize);
                    char* byteArray = (char*)compressedImageMsg->ImageBuffer.data();
                    _imageFile.read(byteArray, imageSize);
                    fileOffset2 = _imageFile.tellg();
                    delFileOffset = (int)(fileOffset2 - fileOffset1);
                    error = delFileOffset != imageSize;
                    if(error)
                    {
                        LOGWARN("ImagePlusMetadataReader:ReadCompressedImage error: at offset: "
                                        << fileOffset1
                                        << " Image Size: " << imageSize
                                        << " DelFileSize: " << delFileOffset);
                    }
                }
            }
            catch (std::exception &e)
            {
                LOGERROR("ImagePlusMetadataReader:ReadCompressedImage: Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }

    bool ImagePlusMetadataReader::ReadRawOpenCVMatImage(cv::Mat *imgMatPtr)
    {
        bool error = true;
        size_t fileOffset1 = 0;
        size_t fileOffset2 = 0;
        int delFileOffset = 0;
        if(_imageFile.is_open())
        {
            try
            {
                fileOffset1 = _imageFile.tellg();

                if (GetNumberOfBytesRemainingInFile() >= _imageHeader.ImageSize)
                {
                    //Check the Mat Size and resize if necessary.
                    imgMatPtr->create(_imageHeader.cvMatRows, _imageHeader.cvMatCols, _imageHeader.cvMatElementType);
                    int rowSize = _imageHeader.cvMatCols * _imageHeader.cvMatElementSize;
                    for(int r = 0; r < _imageHeader.cvMatRows; r++)
                    {
                        char* rowPtr = (char*)imgMatPtr->ptr(r);
                        _imageFile.read(rowPtr, rowSize);
                    }
                    fileOffset2 = _imageFile.tellg();
                    delFileOffset = (int)(fileOffset2 - fileOffset1);
                    error = delFileOffset != _imageHeader.ImageSize;
                    if(error)
                    {
                        LOGWARN("ImagePlusMetadataReader: Error reading raw openCV Mat image, file: "
                                        << _currentFilePath << " Byte Offset: " << fileOffset1);
                    }
                }
            }
            catch (std::exception &e)
            {
                LOGERROR("ImagePlusMetadataReader:ReadImageHeader: Exception: " << e.what());
                error = true;
            }
        }
        return error;
    }

}