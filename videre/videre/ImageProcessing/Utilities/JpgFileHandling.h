/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: March 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#ifndef VIDERE_DEV_JPGFILEHANDLING_H
#define VIDERE_DEV_JPGFILEHANDLING_H

#include "global_defines.h"
#include <opencv2/core.hpp>
#include "../../Utils/logger.h"
#include <string>



namespace VidereImageprocessing
{

    class JpgFileHandler
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::vector<int> _compress_params;

        int _jpgQuality;   //A value between 1 and 100... 100 is the best quality.


    public:
        std::string FileDirectory;
        std::string Filename;

        //Compressed Image buffer
        std::vector<unsigned char> CompressedImageBuffer;

    public:
        JpgFileHandler();

        JpgFileHandler(int jpgQualityFactor);

        ~JpgFileHandler(){}

        void SetJPGQualityFactor(int qf);

        int GetJPGQualityFactor()
        {
            return _jpgQuality;
        }

        //Clear/reset an memory used... CompressedImageBuffer
        void Reset();

        //Compress a cvMat Image to JPEG format...
        //The image is stored in the CompressedImageBuffer
        //returns false if ok.. true if there is an error compressing the image.
        bool CompressImage(const cv::Mat &cvMatImage);

        //The compressed image is stored in the CompressedImageBuffer
        bool StoreCompessedImage(const std::string &filename);

        bool CompressAndStoreImage(const cv::Mat &cvMatImage, const std::string &filename);

        bool ReadImageFromFile(cv::Mat &cvMatImage, const std::string &filename);

    };

}


#endif //VIDERE_DEV_JPGFILEHANDLING_H
