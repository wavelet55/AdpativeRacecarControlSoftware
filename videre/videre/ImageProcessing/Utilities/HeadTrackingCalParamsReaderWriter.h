/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Aug. 2018
 *
  *******************************************************************/

#ifndef VIDERE_DEV_HEADTRACKINGCALPARAMSREADERWRITER_H
#define VIDERE_DEV_HEADTRACKINGCALPARAMSREADERWRITER_H

#include "CommonImageProcTypesDefs.h"
#include "CameraCalibrationData.h"
#include "config_data.h"
#include <boost/filesystem.hpp>
#include "config_data.h"

namespace videre
{


    void ReadHeadTrackingParametersFromIniFile(std::string filename, ImageProcLibsNS::HeadTrackingParameters_t &htParams);

    void ReadHeadTrackingParametersFromConfig(std::shared_ptr<ConfigData> cfg, ImageProcLibsNS::HeadTrackingParameters_t &htParams);

    void WriteHeadTrackingParametersToIniFile(std::string filename, ImageProcLibsNS::HeadTrackingParameters_t &htParams);

    void WriteHeadTrackingParametersToFile(std::ofstream &cfgFile, ImageProcLibsNS::HeadTrackingParameters_t &htParams);


    void ReadHeadOrientationCalDataFromIniFile(std::string filename, ImageProcLibsNS::HeadOrientationCalData_t &calData);

    void ReadHeadOrientationCalDataFromConfig(std::shared_ptr<ConfigData> cfg, ImageProcLibsNS::HeadOrientationCalData_t &calData);

    void WriteHeadOrientationCalDataToIniFile(std::string filename, ImageProcLibsNS::HeadOrientationCalData_t &calData);

    void WriteHeadOrientationCalDataToFile(std::ofstream &cfgFile, ImageProcLibsNS::HeadOrientationCalData_t &calData);


    void ReadHeadModelFromIniFile(std::string filename, std::vector<cv::Point3f> &headModelData);

    void ReadHeadModelFromConfig(std::shared_ptr<ConfigData> cfg, std::vector<cv::Point3f> &headModelData);

}
#endif //VIDERE_DEV_HEADTRACKINGCALPARAMSREADERWRITER_H
