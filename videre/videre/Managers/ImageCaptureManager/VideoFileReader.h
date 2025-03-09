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

#ifndef VIDERE_DEV_VIDEOFILEREADER_H
#define VIDERE_DEV_VIDEOFILEREADER_H


#include "ImageCaptureAClass.h"
#include "RecorderPlayer/ImagePlusMetadataReader.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

namespace videre
{

//Capture images from a webcam using the OpenCV
//library.
    class VideoFileReader : public ImageCaptureAClass
    {
    private:
        std::string _imageDirname = "VideoFiles";
        std::string _videoFilename = "video.avi";

        boost::filesystem::path _currentFilePath; /* Full path to current video */
        std::ifstream _imageFile;
        size_t _imageFileSize = 0;


        std::unique_ptr<cv::VideoCapture> _capture_sptr; /* pull images from this */
        int _numberOfFrames = 0;
        int _imageWidth = 0;
        int _imageHeight = 0;
        int _frameIdx = 0;

        CompressedImageMessage _compressedImageMsg;

    public:
        VideoFileReader(std::shared_ptr<ConfigData> config,
                                  std::shared_ptr<ImageCaptureControlMessage> imageCaptureControlMsg)
                : ImageCaptureAClass(config),
                  _compressedImageMsg()
        {

        }

        ~VideoFileReader()
        {
            Close();
        }

        int GetNumberOfImages()
        {
            return _numberOfFrames;
        }

        bool IsEndOfImages()
        {
            return _frameIdx >= _numberOfFrames;
        }


        //Initalize the Image Capture process.
        //Returns true if error, false if ok.
        virtual bool Initialize(ImageCaptureControlMessage &imgCapCtrlMsg);

        //Initalize the Image Capture process.
        //Returns true if error, false if ok.
        virtual bool Initialize();

        virtual ImageCaptureReturnType_e GetNextImage(ImagePlusMetadataMessage *imgPMetadataMsgPtr);

        //Close the Image Capture Process.
        virtual void Close();

    private:


    };

}

#endif //VIDERE_DEV_VIDEOFILEREADER_H
