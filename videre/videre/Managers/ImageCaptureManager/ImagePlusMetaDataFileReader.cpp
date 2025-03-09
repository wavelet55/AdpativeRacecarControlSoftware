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

#include "ImagePlusMetaDataFileReader.h"

using namespace std;
using namespace VidereImageprocessing;

namespace videre
{

    //Initalize the Image Capture process.
    //Returns true if error, false if ok.
    bool ImagePlusMetaDataFileReader::Initialize()
    {
        ImageCaptureControlMessage imgCapCtrlMsg;
        imgCapCtrlMsg.Clear();
        imgCapCtrlMsg.ImageCaptureSource = ImageCaptureSource_e ::ImageCaptureSource_CImageFiles;

        imgCapCtrlMsg.ImageCaptureSourceConfigPri = _config_sptr->GetConfigStringValue("ImagePlusMetadataReader.directory", "RecordedImages");
        LoopBackToStartOfImages = _config_sptr->GetConfigBoolValue("ImagePlusMetadataReader.LoopImages", "false");
        imgCapCtrlMsg.ImageSourceLoopAround = LoopBackToStartOfImages;
        return Initialize(imgCapCtrlMsg);
    }


    bool
    ImagePlusMetaDataFileReader::Initialize(ImageCaptureControlMessage &imgCapCtrlMsg)
    {
        bool error = true;
        string directory = imgCapCtrlMsg.ImageCaptureSourceConfigPri;
        LOGINFO("Initializing the ImagePlusMetaDataFileReader");
        _ipmReader.resetToStart();
        _numberOfImagesCaptured = 0;
        SetLoopBackToStartOfImages(imgCapCtrlMsg.ImageSourceLoopAround);

        //Start with copying the control message over to the status message...
        //the status message will be updated as needed by actual values.
        ImageCaptureControlStatusMsg.CopyMessage(&imgCapCtrlMsg);
        ImageCaptureControlStatusMsg.ImageCaptureSource = ImageCaptureSource_e ::ImageCaptureSource_IPMFiles;
        ImageCaptureControlStatusMsg.ImageCaptureFormat = ImageCaptureFormat_e ::Unknown;
        //Use the ImageCaptureEnabled flag to indicate the webcam is
        //setup ok.
        ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
        ImageCaptureControlStatusMsg.SetTimeNow();

        try
        {
            int nofiles = _ipmReader.GetListOfFilesFromDirectory(directory);
            if (nofiles > 0)
            {
                LOGINFO("ImagePlusMetaDataFileReader is open. Number of files: " << nofiles);
                _IsImageCaptureInitialized = true;
                error = false;
                ImageCaptureControlStatusMsg.NumberOfImagesToCapture = nofiles;
                ImageCaptureControlStatusMsg.ImageCaptureEnabled = true;
            } else
            {
                LOGINFO("ImagePlusMetaDataFileReader: No Image Plus Metadata files found in dirctory: "
                                << directory << " with file extention: " << _ipmReader.GetIPMFileExtention());
                _IsImageCaptureInitialized = false;
                error = true;
                ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("ImagePlusMetaDataFileReader:Initialize Exception: " << e.what());
            _IsImageCaptureInitialized = false;
            error = true;
            ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
        }
        return error;
    }


    ImageCaptureReturnType_e ImagePlusMetaDataFileReader::GetNextImage(ImagePlusMetadataMessage *imgPMetadataMsgPtr)
    {
        ImageCaptureReturnType_e rtn = ImageCaptureReturnType_e::ICRT_Error;
        ImageReturnType_e irt;
        int decodeFlags;
        try
        {
            if (MaxNumberOfImagesToCapture == 0 || _numberOfImagesCaptured < MaxNumberOfImagesToCapture)
            {
                irt = _ipmReader.ReadNextImagePlusMetadata(imgPMetadataMsgPtr, &_compressedImageMsg);
                imgPMetadataMsgPtr->SetTimeNow();
                switch(irt)
                {
                    case IRT_EndOfImages:
                        imgPMetadataMsgPtr->ImageNumber = 0;
                        _capture_width = 0;
                        _capture_height = 0;
                        rtn = ImageCaptureReturnType_e::IRCT_EndOfImages;
                        break;
                    case IRT_MetadataOnly:
                        ++_imageNumberCounter;
                        ++_numberOfImagesCaptured;
                        imgPMetadataMsgPtr->ImageNumber = _imageNumberCounter;
                        _capture_width = 0;
                        _capture_height = 0;
                        rtn = ImageCaptureReturnType_e::IRCT_MetadataOnly;
                        break;
                    case IRT_MD_RawImage:
                        ++_imageNumberCounter;
                        ++_numberOfImagesCaptured;
                        imgPMetadataMsgPtr->ImageNumber = _imageNumberCounter;
                        //The de-compressed image is in the imgPMetadataMsgPtr
                        SetImageCaptureTime(imgPMetadataMsgPtr->ImageCaptureTimeStampSec);
                        _capture_width = imgPMetadataMsgPtr->ImageFrame.cols;
                        _capture_height = imgPMetadataMsgPtr->ImageFrame.rows;
                        rtn = ImageCaptureReturnType_e::IRCT_ImagePlusMetadata;
                        break;
                    case IRT_MD_CompressedImgage:
                        //Decompress Image
                        decodeFlags = cv::IMREAD_UNCHANGED;
                        cv::imdecode(_compressedImageMsg.ImageBuffer, decodeFlags, &imgPMetadataMsgPtr->ImageFrame);

                        ++_imageNumberCounter;
                        ++_numberOfImagesCaptured;
                        imgPMetadataMsgPtr->ImageNumber = _imageNumberCounter;
                        imgPMetadataMsgPtr->ImageNoPixelsWide = imgPMetadataMsgPtr->ImageFrame.cols;
                        imgPMetadataMsgPtr->ImageNoPixelsHigh = imgPMetadataMsgPtr->ImageFrame.rows;
                        SetImageCaptureTime(imgPMetadataMsgPtr->ImageCaptureTimeStampSec);
                        _capture_width = imgPMetadataMsgPtr->ImageFrame.cols;
                        _capture_height = imgPMetadataMsgPtr->ImageFrame.rows;
                        if(_capture_width < 1 || _capture_height < 1)
                        {
                            rtn = ImageCaptureReturnType_e::ICRT_Error;
                        }
                        else
                        {
                            rtn = ImageCaptureReturnType_e::IRCT_ImagePlusMetadata;

                        }
                        break;
                    case IRT_Error:
                        _capture_width = 0;
                        _capture_height = 0;
                        rtn = ImageCaptureReturnType_e::ICRT_Error;
                        break;
                }

            }
            else
            {
                rtn = ImageCaptureReturnType_e::IRCT_EndOfImages;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("ImagePlusMetaDataFileReader:GetNextImage Exception: " << e.what());
            rtn = ImageCaptureReturnType_e::ICRT_Error;
        }
        return rtn;
    }

    //Close the Image Capture Process.
    void ImagePlusMetaDataFileReader::Close()
    {
        try
        {
            _ipmReader.closeImageFile();
            _IsImageCaptureInitialized = false;
            ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
        }
        catch (std::exception &e)
        {
            LOGERROR("ImagePlusMetaDataFileReader:Close Exception: " << e.what());
        }
    }

}