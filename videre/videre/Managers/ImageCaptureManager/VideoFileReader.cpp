/* ****************************************************************
* DireenTech Inc.  (www.direentech.com)
* Athr: Harry Direen PhD
* Date: Jan 2018
*
* Developed under contract for:
* Academy Center for UAS Research
* Department of Electrical and Computer Engineering
        * HQ USAFA/DFEC
* 2354 Fairchild Drive
* USAF Academy, CO 80840-6236
*
*******************************************************************/

#include "VideoFileReader.h"
#include "FileUtils.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace VidereImageprocessing;

namespace videre
{


    //Initalize the Image Capture process.
    //Returns true if error, false if ok.
    bool VideoFileReader::Initialize()
    {
        ImageCaptureControlMessage imgCapCtrlMsg;
        imgCapCtrlMsg.Clear();
        imgCapCtrlMsg.ImageCaptureSource = ImageCaptureSource_e ::ImageCaptureSource_CImageFiles;

        imgCapCtrlMsg.ImageCaptureSourceConfigPri = _config_sptr->GetConfigStringValue("VideoFileReader.directory", "VideoFiles");
        imgCapCtrlMsg.ImageCaptureSourceConfigSec = _config_sptr->GetConfigStringValue("VideoFileReader.VideoFile", "video.avi");
        LoopBackToStartOfImages = _config_sptr->GetConfigBoolValue("VideoFileReader.LoopVideo", "true");
        imgCapCtrlMsg.ImageSourceLoopAround = LoopBackToStartOfImages;
        return Initialize(imgCapCtrlMsg);
    }

    bool VideoFileReader::Initialize(ImageCaptureControlMessage &imgCapCtrlMsg)
    {
        bool error = true;
        LOGINFO("Initializing the VideoFileReader");
        _imageDirname = imgCapCtrlMsg.ImageCaptureSourceConfigPri;
        _videoFilename = imgCapCtrlMsg.ImageCaptureSourceConfigSec;
        _numberOfFrames = 0;
        _frameIdx = 0;
        _numberOfImagesCaptured = 0;
        LoopBackToStartOfImages = imgCapCtrlMsg.ImageSourceLoopAround;

        //Start with copying the control message over to the status message...
        //the status message will be updated as needed by actual values.
        ImageCaptureControlStatusMsg.CopyMessage(&imgCapCtrlMsg);
        ImageCaptureControlStatusMsg.ImageCaptureSource = ImageCaptureSource_e ::ImageCaptureSource_VideoFile;
        ImageCaptureControlStatusMsg.ImageCaptureFormat = ImageCaptureFormat_e ::Unknown;
        //Use the ImageCaptureEnabled flag to indicate the webcam is
        //setup ok.
        ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
        ImageCaptureControlStatusMsg.SetTimeNow();

        try
        {
            //Full path to video file
            _currentFilePath = _imageDirname;
            _currentFilePath /= _videoFilename;

            //OpenCV is used to capture images from the video file.
            _capture_sptr = unique_ptr<cv::VideoCapture>(new cv::VideoCapture(_currentFilePath.string()));

            if (!_capture_sptr->isOpened())
                _capture_sptr->open(_currentFilePath.string());
            if (!_capture_sptr->isOpened())
            {
                _numberOfFrames = 0;
                ImageCaptureControlStatusMsg.NumberOfImagesToCapture = 0;
                ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;

                LOGERROR("VideoFileReader: Can't open or process video file: " << _currentFilePath);
                _IsImageCaptureInitialized = false;
                error = true;
            }
            else
            {
                //Get the Number of Frames in the video file
                //and the image size info.
                _numberOfFrames = _capture_sptr->get(cv::CAP_PROP_FRAME_COUNT);
                ImageCaptureControlStatusMsg.NumberOfImagesToCapture = _numberOfFrames;
                _imageWidth = _capture_sptr->get(cv::CAP_PROP_FRAME_WIDTH);
                _imageHeight = _capture_sptr->get(cv::CAP_PROP_FRAME_HEIGHT);
                ImageCaptureControlStatusMsg.ImageCaptureEnabled = true;

                LOGINFO("VideoFileReader is open. File: " << _currentFilePath );
                LOGINFO("VideoFile number of frames: " << _numberOfFrames);
                LOGINFO("VideoFile Image Width: " << _imageWidth);
                LOGINFO("VideoFile Image Height: " << _imageHeight);
                _IsImageCaptureInitialized = true;
                error = false;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("VideoFileReader:Initialize Exception: " << e.what());
            _IsImageCaptureInitialized = false;
            error = true;
        }
        return error;
    }



    ImageCaptureReturnType_e VideoFileReader::GetNextImage(ImagePlusMetadataMessage *imgPMetadataMsgPtr)
    {
        ImageCaptureReturnType_e rtn = ImageCaptureReturnType_e::ICRT_Error;
        try
        {
            if(imgPMetadataMsgPtr != NULL )
            {
                if(_frameIdx >= _numberOfFrames && LoopBackToStartOfImages)
                {
                    //reset to the start of the video.
                    //ToDo: Later this process could be used to set the
                    //video to any point in the video.
                    _capture_sptr->set(cv::CAP_PROP_POS_FRAMES, 0);
                    _frameIdx = 0;
                }
                if ( (MaxNumberOfImagesToCapture == 0 || _numberOfImagesCaptured < MaxNumberOfImagesToCapture)
                        && _frameIdx < _numberOfFrames )
                {
                    //Retrieve the image grabbed/captured above;
                    _capture_sptr->read(imgPMetadataMsgPtr->ImageFrame);
                    if (imgPMetadataMsgPtr->ImageFrame.data != NULL
                        && imgPMetadataMsgPtr->ImageFrame.rows > 10
                        && imgPMetadataMsgPtr->ImageFrame.cols > 10)
                    {
                        ++_frameIdx;
                        ++_imageNumberCounter;
                        ++_numberOfImagesCaptured;
                        imgPMetadataMsgPtr->ImageCaptureTimeStampSec = GetImageCaptureTime();
                        imgPMetadataMsgPtr->SetTimeNow();
                        imgPMetadataMsgPtr->ImageNumber = _imageNumberCounter;
                        imgPMetadataMsgPtr->ImageNoPixelsWide = imgPMetadataMsgPtr->ImageFrame.cols;
                        imgPMetadataMsgPtr->ImageNoPixelsHigh = imgPMetadataMsgPtr->ImageFrame.rows;
                        rtn = ImageCaptureReturnType_e::IRCT_ImageOnly;
                    }
                    else
                    {
                        imgPMetadataMsgPtr->ImageCaptureTimeStampSec = GetImageCaptureTime();
                        imgPMetadataMsgPtr->SetTimeNow();
                        imgPMetadataMsgPtr->ImageNumber = 0;
                        imgPMetadataMsgPtr->ImageNoPixelsWide = 0;
                        imgPMetadataMsgPtr->ImageNoPixelsHigh = 0;
                        rtn = ImageCaptureReturnType_e::IRCT_EndOfImages;
                    }
                }
                else
                {
                    rtn = ImageCaptureReturnType_e::IRCT_EndOfImages;
                }
            }
            else
            {
                rtn = ImageCaptureReturnType_e::IRCT_EndOfImages;
            }
        }
        catch(std::exception &e)
        {
            LOGERROR("OpenCVWebcamImageCapture:GetNextImage Exception: " << e.what());
            rtn = ImageCaptureReturnType_e::ICRT_Error;
        }
        return rtn;

    }

    //Close the Image Capture Process.
    void VideoFileReader::Close()
    {
        try
        {
            if(_capture_sptr != nullptr)
            {
                _capture_sptr->release();
                _IsImageCaptureInitialized = false;
                ImageCaptureControlStatusMsg.ImageCaptureEnabled = false;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("VideoFileReader:Close Exception: " << e.what());
        }
    }

}
