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

#ifndef VIDERE_DEV_IMAGEPLUSMETADATARECORDER_H
#define VIDERE_DEV_IMAGEPLUSMETADATARECORDER_H

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "global_defines.h"
#include <opencv2/core.hpp>
#include "LatLonAltStruct.h"
#include "XYZCoord_t.h"
#include "RollPitchYaw_t.h"
#include "AzimuthElevation_t.h"
#include "config_data.h"
#include "../../Utils/logger.h"
#include "ImagePlusMetadataFileHeaders.h"
#include <boost/filesystem.hpp>

using namespace videre;
using namespace GeoCoordinateSystemNS;
using namespace MathLibsNS;

namespace VidereImageprocessing
{
    /*****************************
     * Desc: Image Storage
     * This class combines images with their metadata (vehical position,
     * camera orientation, time, etc) into a single file.  Multiple images
     * along with their individual metadata will be combined into a single
     * file.  The operating system works more efficiently with fewer large
     * files, than a large number of smaller files.  Once a file grows to
     * a given size, the file will be closed out, and a new file started.
     *
     * A file header will start each Image file. The file header will provide
     * basic information about the Image File, the Vehicle, and the Mission.
     * This information will be in text format.
     *
     * Image Meta-data will be stored in binary format rather than in text
     * format with markup such as an xml format, as the binary format is
     * more efficient.
     */
    class ImagePlusMetadataRecorder
    {
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        std::shared_ptr<ConfigData> _config_sptr;

        const int MegaByte = 1024 * 1024;
        const std::string IPM_EXTENSION_STR = "imd";

        bool _imageDirExists = false;
        bool _imageSaveError = false;
        int _currentFileNumber = 0;
        int _rawImageFileNo = 0;
        ImagePlusMetadataFileHeader _imageFileHeader;
        ImagePlusMetadataHeader _imageHeader;
        std::string _imageDirname;
        std::string _baseFilename; /* Base filename*/
        boost::filesystem::path _currentFilePath; /* Full path to current video */
        std::ofstream _imageFile;

        size_t MaxFileSize = 100 * MegaByte;

    public:
        bool SaveImages = true;   //Disable to save metadata only

    public:
        ImagePlusMetadataRecorder(std::shared_ptr<ConfigData> config,
                                  bool createDefaultDirectory);

        ~ImagePlusMetadataRecorder();

        bool GetImageSaveError()
        {
            return _imageSaveError;
        }


        bool CreateDirectoryFromCfgDirname(bool addTimestamp);

        bool CreateDirectory(const std::string &directory, bool addTimestamp);

        bool openNewImageFile();

        void closeImageFile();

        bool WriteImagePlusMetadata(ImagePlusMetadataMessage &imgMdMsg,
                                    ImageStorageType_e imgType,
                                    char* imageArray = nullptr,
                                    int imageArraySize = 0,
                                    ImageFormatType_e imageFormatType = ImageFormatType_e::ImgFType_Raw);

    private:

        //This is based upon the file pointer being at the
        //end of the file... works when writting to the file at the end.
        size_t GetCurrentFileSize()
        {
            size_t size = 0;
            if(_imageFile.is_open())
            {
                size = _imageFile.tellp();
            }
            return size;
         }
    };

}


#endif //VIDERE_DEV_IMAGEPLUSMETADATARECORDER_H
