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

#include "JpgFileHandling.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "FileUtils.h"
#include <iostream>
#include <fstream>

using namespace videre;
using namespace std;
using namespace cv;

namespace VidereImageprocessing
{

    JpgFileHandler::JpgFileHandler()
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        SetJPGQualityFactor(90);
    }

    JpgFileHandler::JpgFileHandler(int jpgQualityFactor)
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        SetJPGQualityFactor(jpgQualityFactor);
    }

    void JpgFileHandler::SetJPGQualityFactor(int qf)
    {
        _jpgQuality = qf < 1 ? 1 : qf > 100 ? 100 : qf;
        _compress_params.clear();
        _compress_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        _compress_params.push_back(_jpgQuality);
    }

    //Clear/reset an memory used... CompressedImageBuffer
    void JpgFileHandler::Reset()
    {
        CompressedImageBuffer.clear();
        CompressedImageBuffer.shrink_to_fit();
    }

    //Compress a cvMat Image to JPEG format...
    //The image is stored in the CompressedImageBuffer
    //returns false if ok.. true if there is an error compressing the image.
    bool JpgFileHandler::CompressImage(const cv::Mat &cvMatImage)
    {
        bool error = true;
        try
        {
            CompressedImageBuffer.clear();
            //Compress the Image before recording or transmitting the image.
            error = !imencode(".jpg", cvMatImage, CompressedImageBuffer, _compress_params);
        }
        catch (std::exception &e)
        {
            LOGERROR("JpgFileHandler.CompressImage: Exception: " << e.what());
            error = true;
        }
        return error;
    }


    //The compressed image is stored in the CompressedImageBuffer
    bool JpgFileHandler::StoreCompessedImage(const std::string &filename)
    {
        bool error = true;
        std::ofstream _imageFile;
        ios_base::openmode fileMode = ios_base::out | ios_base::trunc | ios_base::binary;
        Filename = filename;
        try
        {
            if(CompressedImageBuffer.size() > 100)
            {
                _imageFile.open(filename, fileMode);
                _imageFile.write((char*)CompressedImageBuffer.data(),
                                 CompressedImageBuffer.size());
                _imageFile.close();
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("JpgFileHandler.StoreCompessedImage: Exception: " << e.what());
            _imageFile.close();
            error = true;
        }
        return error;
    }

    bool JpgFileHandler::CompressAndStoreImage(const cv::Mat &cvMatImage, const std::string &filename)
    {
        bool error = true;
        error = CompressImage(cvMatImage);
        if(!error)
        {
            error = StoreCompessedImage(filename);
        }
        return error;
    }

    bool JpgFileHandler::ReadImageFromFile(cv::Mat &cvMatImage, const std::string &filename)
    {
        bool error = true;
        std::ifstream _imageFile;
        ios_base::openmode fileMode = ios_base::in | ios_base::binary;
        Filename = filename;
        try
        {
            _imageFile.open(filename, fileMode);

            _imageFile.seekg (0, std::ios::end);
            size_t imageFileSize = _imageFile.tellg();
            _imageFile.seekg (0, std::ios::beg);

            CompressedImageBuffer.resize(imageFileSize);
            char* byteArray = (char*)CompressedImageBuffer.data();
            _imageFile.read(byteArray, imageFileSize);
            _imageFile.close();

            //Decompress Image
            int decodeFlags = cv::IMREAD_UNCHANGED;
            cv::imdecode(CompressedImageBuffer, decodeFlags, &cvMatImage);
            error = false;
        }
        catch (std::exception &e)
        {
            LOGERROR("JpgFileHandler.ReadImageFromFile: Exception: " << e.what());
            _imageFile.close();
            error = true;
        }
        return error;
    }


}