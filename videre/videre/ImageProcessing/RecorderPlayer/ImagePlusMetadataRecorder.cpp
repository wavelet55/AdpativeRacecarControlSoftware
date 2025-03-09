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


#include "ImagePlusMetadataRecorder.h"
#include "FileUtils.h"
#include "boost/date_time/gregorian/gregorian.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
#include <iostream>

using namespace std;

namespace VidereImageprocessing
{

    ImagePlusMetadataRecorder::ImagePlusMetadataRecorder(std::shared_ptr<ConfigData> config,
                    bool createDefaultDirectory)
        :_imageFileHeader(),
         _imageHeader(),
         _imageDirname("ImageDir")
    {
        _config_sptr = config;
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        SaveImages = false;
        _imageDirExists = false;
        _imageSaveError = false;
        _currentFileNumber = 0;

        _baseFilename = _config_sptr->GetConfigStringValue("video_recording.base_name", "ImagePlusMetadata");

        int maxFSMB = _config_sptr->GetConfigIntValue("video_recording.max_filesize_megabytes", 100);
        MaxFileSize = maxFSMB * MegaByte;

        //This does not change... so create it once and use the
        //results when creating new files.
        _imageFileHeader.CreateFileHeader();

        if(createDefaultDirectory )
        {
            CreateDirectoryFromCfgDirname(true);
        }

    }

    ImagePlusMetadataRecorder::~ImagePlusMetadataRecorder()
    {
        closeImageFile();
    }

    bool ImagePlusMetadataRecorder::CreateDirectoryFromCfgDirname(bool addTimestamp)
    {
        string dirname = _config_sptr->GetConfigStringValue("video_recording.directory", "RecordedImages");
        return CreateDirectory(dirname, addTimestamp);
    }

    bool ImagePlusMetadataRecorder::CreateDirectory(const std::string &directory, bool addTimestamp)
    {
        _imageDirname = directory;
        _imageDirExists = false;
        try
        {
            if (addTimestamp)
            {
                _imageDirname = VidereFileUtils::AddCurrentTimeDateStampToString(_imageDirname);
            }
            if (VidereFileUtils::CreateDirectory(_imageDirname))
            {
                _imageDirExists = true;
            }
            else
            {
                LOGERROR("ImagePlusMetadataRecorder:CreateDirectory: Could not create direcory: " << _imageDirname);
                _imageDirExists = false;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("ImagePlusMetadataRecorder:CreateDirectory: Exception: " << e.what());
        }
        return _imageDirExists;
    }

    bool ImagePlusMetadataRecorder::openNewImageFile()
    {
        bool error = true;
        try
        {
            //First ensure last file is closed
            if(_imageFile.is_open())
            {
                _imageFile.close();
            }

            if(!_imageDirExists)
            {
                if(!CreateDirectoryFromCfgDirname(true))
                {
                    return true;
                }
            }

            ++_currentFileNumber;
            string filename = VidereFileUtils::AddIndexToFilename(_baseFilename,
                                                         _currentFileNumber,
                                                         4, IPM_EXTENSION_STR);
            _currentFilePath = _imageDirname + "/" + filename;
            ios_base::openmode fileMode = ios_base::out | ios_base::trunc | ios_base::binary;
            _imageFile.open(_currentFilePath.c_str(), fileMode);

            //Write File Header to the file.
            _imageFile.write(_imageFileHeader.GetFileHeaderArray(),
                             _imageFileHeader.FileHeaderSize);

            error = false;
        }
        catch (std::exception &e)
        {
            LOGERROR("ImagePlusMetadataRecorder:openNewImageFile: Exception: " << e.what());
            error = true;
        }
        return error;
    }

    void ImagePlusMetadataRecorder::closeImageFile()
    {
        try
        {
            if(_imageFile.is_open())
            {
                _imageFile.close();
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("ImagePlusMetadataRecorder:closeImageFile: Exception: " << e.what());
        }
    }

    bool ImagePlusMetadataRecorder::WriteImagePlusMetadata(ImagePlusMetadataMessage &imgMdMsg,
                                                           ImageStorageType_e imgType,
                                                           char* imageArray,
                                                           int imageArraySize,
                                                           ImageFormatType_e imageFormatType)
    {
        bool error = true;
        if(!_imageFile.is_open())
        {
            if( openNewImageFile() )
            {
                LOGERROR("ImagePlusMetadataRecorder:WriteImagePlusMetadata: Can't open Image File.");
                return true;
            }
        }
        try
        {
            _imageHeader.ImageStorageType = imgType;
            _imageHeader.ImageFormatType = ImageFormatType_e::ImgFType_Raw;
            switch (imgType)
            {
                case IST_None:
                    _imageHeader.ImageSize = 0;
                    break;
                case IST_JPEG:
                    if (imageArraySize > 0 && imageArray != nullptr)
                    {
                        _imageHeader.ImageSize = imageArraySize;
                        _imageHeader.ImageFormatType = imageFormatType;
                    }
                    else
                    {
                        LOGWARN("ImagePlusMetadataRecorder:WriteImagePlusMetadata JPEG: No Image");
                        _imageHeader.ImageSize = 0;
                    }
                    break;
                case IST_OpenCVMatRGB:
                case IST_OpenCVMatLum:
                    _imageHeader.ImageSize = _imageHeader.SetCVMatParamsComputeMatSize(imgMdMsg);
                   break;
                default:
                    _imageHeader.ImageSize = 0;
                    break;
            }

            char *imgHdr = _imageHeader.CreateImageHeader(imgMdMsg);
            _imageFile.write(imgHdr, _imageFileHeader.ImageHeaderSize);

            if (_imageHeader.ImageSize > 0)
            {
                switch (imgType)
                {
                    case IST_None:
                        break;
                    case IST_JPEG:
                        _imageFile.write(imageArray, _imageHeader.ImageSize);
                        break;
                    case IST_OpenCVMatRGB:
                    case IST_OpenCVMatLum:
                        //Note:  this is platform (Endianess) dependent... as
                        //the cvMat data structure is Endianess dependent
                        int rowSize = _imageHeader.cvMatCols * _imageHeader.cvMatElementSize;
                        for(int r = 0; r < _imageHeader.cvMatRows; r++)
                        {
                            char* rowPtr = (char*)imgMdMsg.ImageFrame.ptr(r);
                            _imageFile.write(rowPtr, rowSize);
                        }
                        break;
                }
            }
            if( GetCurrentFileSize() >= MaxFileSize)
            {
                closeImageFile();
            }
            error = false;
        }
        catch (std::exception &e)
        {
            LOGERROR("ImagePlusMetadataRecorder:WriteImagePlusMetadata: Exception: " << e.what());
        }
        return error;
    }


}
