/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: March 2017
 *
 *  Routines to read and write IMU calibration data files and read
 *  and write cal data to IMU.
 *
 *  Calibration is handled according to the reference:
 *  "A Robust and Easy to Implement Method for IMU Calibration without
 *  External Equipments" by: David Tedaldi, Alberto Pretto and
 *  Emanuele Menegatti
  *******************************************************************/

#include "IMUCalibration.h"
#include "FileUtils.h"
#include <iostream>
#include <fstream>

using namespace std;

namespace IMU_SensorNS
{

    IMUCalibration::IMUCalibration()
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        IMUCalData.Clear();
    }


    void IMUCalibration::ReadAccelerometerIMUCalInfo(ConfigData &calDataCfg)
    {
        std::ostringstream cfgParm;
        IMUCalData_t *cdPtr;
        string cfgPathStr= "IMUCalAccelerometer.";;

        cfgParm.str(std::string());  //Clear the string stream
        cfgParm << cfgPathStr << "Gain_X";
        IMUCalData.AccelerometerGains.x = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 1.0);
        cfgParm.str(std::string());  //Clear the string stream
        cfgParm << cfgPathStr << "Gain_Y";
        IMUCalData.AccelerometerGains.y = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 1.0);
        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Gain_Z";
        IMUCalData.AccelerometerGains.z = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 1.0);

        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Bias_X";
        IMUCalData.AccelerometerBias.x = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Bias_Y";
        IMUCalData.AccelerometerBias.y = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Bias_Z";
        IMUCalData.AccelerometerBias.z = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);

        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Rot_yz";
        IMUCalData.AccelerometerRotationVals.Ryz = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Rot_zy";
        IMUCalData.AccelerometerRotationVals.Rzy = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Rot_zx";
        IMUCalData.AccelerometerRotationVals.Rzx = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
    }

    void IMUCalibration::ReadGyroIMUCalInfo(ConfigData &calDataCfg)
    {
        std::ostringstream cfgParm;
        IMUCalData_t *cdPtr;
        string cfgPathStr = "IMUCalGyro.";;

        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Gain_X";
        IMUCalData.GyroGains.x = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 1.0);
        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Gain_Y";
        IMUCalData.GyroGains.y = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 1.0);
        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Gain_Z";
        IMUCalData.GyroGains.z = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 1.0);

        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Bias_X";
        IMUCalData.GyroBias.x = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Bias_Y";
        IMUCalData.GyroBias.y = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Bias_Z";
        IMUCalData.GyroBias.z = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);

        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Rot_yz";
        IMUCalData.GyroRotationVals.Ryz = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Rot_zy";
        IMUCalData.GyroRotationVals.Rzy = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Rot_zx";
        IMUCalData.GyroRotationVals.Rzx = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);

        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Rot_xz";
        IMUCalData.GyroRotationVals.Rxz = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Rot_xy";
        IMUCalData.GyroRotationVals.Rxy = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
        cfgParm.str(std::string());  //Clear the string stream;
        cfgParm << cfgPathStr << "Rot_yx";
        IMUCalData.GyroRotationVals.Ryx = calDataCfg.GetConfigDoubleValue(cfgParm.str(), 0.0);
    }


    //Read IMU Calibration from the given file.
    //The file name is the full path to the file.
    //throws exception:  ConfigParseException() if the file cannot be read.
    bool IMUCalibration::ReadIMUCalibrationFromIniFile(std::string filename)
    {
        bool error = false;
        ConfigData calDataCfg;
        try
        {
            calDataCfg.ParseConfigFile(filename);
            ReadAccelerometerIMUCalInfo(calDataCfg);
            ReadGyroIMUCalInfo(calDataCfg);
        }
        catch(exception &e)
        {
            LOGWARN( "Error reading IMU Calibration File: " + filename
                       + " Exception: " + e.what() );
            error = true;
        }
        return error;
    }

    //The Accelerometer Calibration Parameters are indexed
    //based on the IMU code:  Gx,Gy,Gz,Bx,By,Bz...
    IMUCalParameter_t IMUCalibration::GetNextAccelerometerCalParameter()
    {
        IMUCalParameter_t calP;
        calP.idx = _calParamIdx;
        switch(_calParamIdx)
        {
            case 0:
                calP.pVal = IMUCalData.AccelerometerGains.x;
                break;
            case 1:
                calP.pVal = IMUCalData.AccelerometerGains.y;
                break;
            case 2:
                calP.pVal = IMUCalData.AccelerometerGains.z;
                break;
            case 3:
                calP.pVal = IMUCalData.AccelerometerBias.x;
                break;
            case 4:
                calP.pVal = IMUCalData.AccelerometerBias.y;
                break;
            case 5:
                calP.pVal = IMUCalData.AccelerometerBias.z;
                break;
            case 6:
                calP.pVal = IMUCalData.AccelerometerRotationVals.Ryz;
                break;
            case 7:
                calP.pVal = IMUCalData.AccelerometerRotationVals.Rzy;
                break;
            case 8:
                calP.pVal = IMUCalData.AccelerometerRotationVals.Rzx;
                break;
            default:
                calP.pVal = 0;
                calP.idx = - 1;
        }
        ++_calParamIdx;
        return calP;
    }

    //The Gyro Calibration Parameters are indexed
    //based on the IMU code:  Gx,Gy,Gz,Bx,By,Bz...
    IMUCalParameter_t IMUCalibration::GetNextGyroCalParameter()
    {
        IMUCalParameter_t calP;
        calP.idx = _calParamIdx;
        switch(_calParamIdx)
        {
            case 0:
                calP.pVal = IMUCalData.GyroGains.x;
                break;
            case 1:
                calP.pVal = IMUCalData.GyroGains.y;
                break;
            case 2:
                calP.pVal = IMUCalData.GyroGains.z;
                break;
            case 3:
                calP.pVal = IMUCalData.GyroBias.x;
                break;
            case 4:
                calP.pVal = IMUCalData.GyroBias.y;
                break;
            case 5:
                calP.pVal = IMUCalData.GyroBias.z;
                break;
            case 6:
                calP.pVal = IMUCalData.GyroRotationVals.Ryz;
                break;
            case 7:
                calP.pVal = IMUCalData.GyroRotationVals.Rzy;
                break;
            case 8:
                calP.pVal = IMUCalData.GyroRotationVals.Rzx;
                break;
            case 9:
                calP.pVal = IMUCalData.GyroRotationVals.Rxz;
                break;
            case 10:
                calP.pVal = IMUCalData.GyroRotationVals.Rxy;
                break;
            case 11:
                calP.pVal = IMUCalData.GyroRotationVals.Ryx;
                break;
            default:
                calP.pVal = 0;
                calP.idx = - 1;
        }
        ++_calParamIdx;
        return calP;
    }


}