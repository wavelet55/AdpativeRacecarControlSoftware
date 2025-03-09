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
  
#ifndef VIDERE_DEV_IMAGEPLUSMETADATAFILEREADER_H
#define VIDERE_DEV_IMAGEPLUSMETADATAFILEREADER_H

#include "ImageCaptureAClass.h"
#include "RecorderPlayer/ImagePlusMetadataReader.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/imagecodecs/imgcodecs.hpp> //OpenCV 3.0 only
//#include <opencv2/videoio/videoio.hpp> //OpenCV 3.0 only
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace videre
{

//Capture images from a webcam using the OpenCV
//library.
    class ImagePlusMetaDataFileReader : public ImageCaptureAClass
    {
    private:
        VidereImageprocessing::ImagePlusMetadataReader _ipmReader;

        CompressedImageMessage _compressedImageMsg;

    public:
        ImagePlusMetaDataFileReader(std::shared_ptr<ConfigData> config,
                                 std::shared_ptr<ImageCaptureControlMessage> imageCaptureControlMsg)
                : ImageCaptureAClass(config),
                  _ipmReader(config),
                  _compressedImageMsg()
        {

        }

        ~ImagePlusMetaDataFileReader()
        {
            Close();
        }

        void SetLoopBackToStartOfImages(bool loopBackFlag)
        {
            _ipmReader.LoopBackToStartOfImages = loopBackFlag;
            LoopBackToStartOfImages = loopBackFlag;
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

    };

}

#endif //VIDERE_DEV_IMAGEPLUSMETADATAFILEREADER_H
