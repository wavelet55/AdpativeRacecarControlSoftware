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

#include "CompressedImageFileReader.h"
#include "FileUtils.h"

using namespace std;
using namespace VidereImageprocessing;

namespace videre
{


    //Initalize the Image Capture process.
    //Returns true if error, false if ok.
    bool CompressedImageFileReader::Initialize()
    {
        ImageCaptureControlMessage imgCapCtrlMsg;
        imgCapCtrlMsg.Clear();
        imgCapCtrlMsg.ImageCaptureSource = ImageCaptureSource_e ::ImageCaptureSource_CImageFiles;

        imgCapCtrlMsg.ImageCaptureSourceConfigPri = _config_sptr->GetConfigStringValue("CompressedImageFileReader.directory", "RecordedImages");
        imgCapCtrlMsg.ImageCaptureSourceConfigSec = _config_sptr->GetConfigStringValue("CompressedImageFileReader.fileExtention", "jpg");
        LoopBackToStartOfImages = _config_sptr->GetConfigBoolValue("CompressedImageFileReader.LoopImages", "true");
        imgCapCtrlMsg.ImageSourceLoopAround = LoopBackToStartOfImages;
        return Initialize(imgCapCtrlMsg);
    }

    bool CompressedImageFileReader::Initialize(ImageCaptureControlMessage &imgCapCtrlMsg)
    {
        bool error = true;
        LOGINFO("Initializing the CompressedImageFileReader");
        _imageDirname = imgCapCtrlMsg.ImageCaptureSourceConfigPri;
        _imageFileExt = imgCapCtrlMsg.ImageCaptureSourceConfigSec;
        _imageFileIndex = 0;
        _numberOfImagesCaptured = 0;
        LoopBackToStartOfImages = imgCapCtrlMsg.ImageSourceLoopAround;

        //Start with copying the control message over to the status message...
        //the status message will be updated as needed by actual values.
        ImageCaptureControlStatusMsg.CopyMessage(&imgCapCtrlMsg);
        ImageCaptureControlStatusMsg.ImageCaptureSource = ImageCaptureSource_e ::ImageCaptureSource_CImageFiles;
        ImageCaptureControlStatusMsg.ImageCaptureFormat = ImageCaptureFormat_e ::Unknown;
        //Use the ImageCaptureEnabled flag to indicate the webcam is
        //setup ok.
        ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
        ImageCaptureControlStatusMsg.SetTimeNow();

        try
        {
            int nofiles = VidereFileUtils::GetListFilesInDirectory(&_listOfImageFiles,
                                                                   _imageDirname,
                                                                   _imageFileExt, "", true);
            if (nofiles > 0)
            {
                LOGINFO("CompressedImageFileReader is open. Number of files: " << nofiles);
                _IsImageCaptureInitialized = true;
                error = false;
                ImageCaptureControlStatusMsg.NumberOfImagesToCapture = nofiles;
                ImageCaptureControlStatusMsg.ImageCaptureEnabled = true;
            } else
            {
                LOGINFO("CompressedImageFileReader: No Image files found in dirctory: "
                                << _imageDirname << " with file extention: " << _imageFileExt);
                _IsImageCaptureInitialized = false;
                error = true;
                ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("CompressedImageFileReader:Initialize Exception: " << e.what());
            _IsImageCaptureInitialized = false;
            error = true;
        }
        return error;
    }

    bool CompressedImageFileReader::ReadImageFromFile(boost::filesystem::path filename,
                                                      CompressedImageMessage *compressedImageMsg)
    {
        bool error = true;
        try
        {
            ios_base::openmode fileMode = ios_base::in | ios_base::binary;
            _imageFile.open(filename.c_str(), fileMode);
            //Ensure we are at the beginning of the file
            _imageFile.seekg (0, std::ios::end);
            _imageFileSize = _imageFile.tellg();
            _imageFile.seekg (0, std::ios::beg);

            compressedImageMsg->ImageBuffer.resize(_imageFileSize);
            char* byteArray = (char*)compressedImageMsg->ImageBuffer.data();
            _imageFile.read(byteArray, _imageFileSize);

            _imageFile.close();
            SetImageCaptureTimeToNow();
            error = false;
        }
        catch (std::exception &e)
        {
            LOGERROR("CompressedImageFileReader:ReadImageFromFile Exception: " << e.what());
            if(_imageFile.is_open())
            {
                _imageFile.close();
            }
            error = true;
        }
        return error;
    }


    ImageCaptureReturnType_e CompressedImageFileReader::GetNextImage(ImagePlusMetadataMessage *imgPMetadataMsgPtr)
    {
        ImageCaptureReturnType_e rtn = ImageCaptureReturnType_e::ICRT_Error;
        ImageReturnType_e irt;
        try
        {
            bool imagesAvailable = _listOfImageFiles.size() > 0;
            if( _imageFileIndex >= _listOfImageFiles.size())
            {
                if(LoopBackToStartOfImages)
                {
                    _imageFileIndex = 0;
                }
                else
                {
                    imagesAvailable = false;
                }
            }
            if (imagesAvailable
                 &&  ( MaxNumberOfImagesToCapture == 0
                        || _numberOfImagesCaptured < MaxNumberOfImagesToCapture) )
            {
                _currentFilePath = _listOfImageFiles[_imageFileIndex];
                ++_imageFileIndex;
                if(ReadImageFromFile(_currentFilePath, &_compressedImageMsg))
                {
                    imgPMetadataMsgPtr->SetTimeNow();
                    imgPMetadataMsgPtr->ImageNumber = 0;
                    _capture_width = 0;
                    _capture_height = 0;
                    imgPMetadataMsgPtr->ImageCaptureTimeStampSec = 0;
                    imgPMetadataMsgPtr->ImageNoPixelsWide = 0;
                    imgPMetadataMsgPtr->ImageNoPixelsHigh = 0;
                    rtn = ImageCaptureReturnType_e::ICRT_Error;
                }
                else
                {
                    //Decompress Image
                    int decodeFlags = cv::IMREAD_UNCHANGED;
                    cv::imdecode(_compressedImageMsg.ImageBuffer, decodeFlags, &imgPMetadataMsgPtr->ImageFrame);

                    ++_imageNumberCounter;
                    ++_numberOfImagesCaptured;
                    imgPMetadataMsgPtr->ImageNumber = _imageNumberCounter;
                    imgPMetadataMsgPtr->SetTimeNow();
                    _capture_width = imgPMetadataMsgPtr->ImageFrame.cols;
                    _capture_height = imgPMetadataMsgPtr->ImageFrame.rows;
                    imgPMetadataMsgPtr->ImageCaptureTimeStampSec = GetImageCaptureTime();
                    imgPMetadataMsgPtr->ImageNoPixelsWide = _capture_width;
                    imgPMetadataMsgPtr->ImageNoPixelsHigh = _capture_height;
                    rtn = ImageCaptureReturnType_e::IRCT_ImageOnly;
                }
            }
            else
            {
                rtn = ImageCaptureReturnType_e::IRCT_EndOfImages;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("CompressedImageFileReader:GetNextImage Exception: " << e.what());
            rtn = ImageCaptureReturnType_e::ICRT_Error;
        }
        return rtn;
    }

    //Close the Image Capture Process.
    void CompressedImageFileReader::Close()
    {
        try
        {
            if(_imageFile.is_open())
            {
                _imageFile.close();
            }
            _IsImageCaptureInitialized = false;
            ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
        }
        catch (std::exception &e)
        {
            LOGERROR("CompressedImageFileReader:Close Exception: " << e.what());
        }
    }

}
