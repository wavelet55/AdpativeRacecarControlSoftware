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
 *  Routines to read and write camera calibration data files.
  *******************************************************************/

#ifndef VIDERE_DEV_CAMERACALREADERWRITER_H
#define VIDERE_DEV_CAMERACALREADERWRITER_H

#include "CommonImageProcTypesDefs.h"
#include "CameraCalibrationData.h"
#include "config_data.h"
#include <boost/filesystem.hpp>
#include "config_data.h"

namespace CameraCalibrationNS
{

    //Read Camera Calibration from the given file.
    //The file name is the full path to the file.
    //The calibration data is stored in the cameraCalData;
    //Calibration files can contain more than one set of calibration data... use
    //calDataIndex for an index to the calibration data.
    //throws exception:  ConfigParseException() if the file cannot be read.
    void ReadCameraCalibrationFromIniFile(std::string filename,
                                       ImageProcLibsNS::CameraCalibrationData & cameraCalData,
                                       int calDataIndex = 0);

    //Read Camera Calibration from the given ini configuration.
    //The file name is the full path to the file.
    //The calibration data is stored in the cameraCalData;
    //Calibration files can contain more than one set of calibration data... use
    //calDataIndex for an index to the calibration data.
    //throws exception:  ConfigParseException() if the file cannot be read.
    void ReadCameraCalibrationDataFromIniConfig(videre::ConfigData &calDataCfg,
                                    ImageProcLibsNS::CameraCalibrationData & cameraCalData,
                                    int calDataIndex);


    //Write the Camera Calibration Data to an ini file.
    //The full path for the file is: filename
    //Throws excecption:  ConfigParseException() if a file cannot be opened or there
    //is errors writing to the file.
    void WriteCameraCalibrationToIniFile(std::string filename,
                                       const ImageProcLibsNS::CameraCalibrationData & cameraCalData);


    //Write the Camera Calibration Data to an ini file.
    //The full path for the file is: filename
    //Throws excecption:  ConfigParseException() if a file cannot be opened or there
    //is errors writing to the file.
    void WriteCameraCalibrationDataToIniFile(std::ofstream &cfgFile,
                                             const ImageProcLibsNS::CameraCalibrationData & cameraCalData,
                                             int calDataIndex = 0);


    class CameraCalReaderWriter
    {

    };

}


#endif //VIDERE_DEV_CAMERACALREADERWRITER_H
