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
  
#ifndef VIDERE_DEV_COMPRESSEDIMAGEFILEREADER_H
#define VIDERE_DEV_COMPRESSEDIMAGEFILEREADER_H

#include "ImageCaptureAClass.h"
#include "RecorderPlayer/ImagePlusMetadataReader.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/imagecodecs/imgcodecs.hpp> //OpenCV 3.0 only
//#include <opencv2/videoio/videoio.hpp> //OpenCV 3.0 only
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

namespace videre
{

//Capture images from a webcam using the OpenCV
//library.
    class CompressedImageFileReader : public ImageCaptureAClass
    {
    private:
        std::string _imageFileExt = "jpg";

        std::string _imageDirname;
        boost::filesystem::path _currentFilePath; /* Full path to current video */
        std::ifstream _imageFile;
        size_t _imageFileSize = 0;

        std::vector<boost::filesystem::path> _listOfImageFiles;
        int _imageFileIndex = 0;


        CompressedImageMessage _compressedImageMsg;

    public:
        CompressedImageFileReader(std::shared_ptr<ConfigData> config,
                                 std::shared_ptr<ImageCaptureControlMessage> imageCaptureControlMsg)
                : ImageCaptureAClass(config),
                   _compressedImageMsg()
        {

        }

        ~CompressedImageFileReader()
        {
            Close();
        }

        int GetNumberOfImages()
        {
            return _listOfImageFiles.size();
        }

        bool IsEndOfImages()
        {
            return _imageFileIndex >= _listOfImageFiles.size();;
        }

        std::string GetFileExtention()
        {
            return _imageFileExt;
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
        //Read an image from the given full file and path name
        //and store the image in the _compressedImageMsg object.
        bool ReadImageFromFile(boost::filesystem::path filename, CompressedImageMessage *_compressedImageMsg);


    };

}


#endif //VIDERE_DEV_COMPRESSEDIMAGEFILEREADER_H
