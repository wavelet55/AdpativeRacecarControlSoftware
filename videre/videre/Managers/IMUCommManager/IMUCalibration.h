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


#ifndef VIDERE_DEV_IMUCALIBRATION_H
#define VIDERE_DEV_IMUCALIBRATION_H

#include "config_data.h"
#include <boost/filesystem.hpp>
#include "config_data.h"
#include "XYZCoord_t.h"
#include "logger.h"
#include <RabitManager.h>

using namespace videre;
using namespace MathLibsNS;

namespace IMU_SensorNS
{

    //Accelerometer Axis Rotation Correction factors.
    struct IMUAccelerometerAxisCorrectionFactors_t
    {
        double Ryz;
        double Rzy;
        double Rzx;

        void Clear()
        {
            Ryz = 0;
            Rzy = 0;
            Rzx = 0;
        }
    };

    //Gyro Axis Rotation Correction factors.
    struct IMUGyroAxisCorrectionFactors_t
    {
        double Ryz;
        double Rzy;
        double Rzx;
        double Rxz;
        double Rxy;
        double Ryx;

        void Clear()
        {
            Ryz = 0;
            Rzy = 0;
            Rzx = 0;
            Rxz = 0;
            Rxy = 0;
            Ryx = 0;
        }
    };

    struct IMUCalParameter_t
    {
        int idx;
        double pVal;
    };

    enum IMUCalParameterSetIdx_e
    {
        ICPS_FixedAccelPSet = 0,
        ICPS_FixedGyroPSet = 1,
        ICPS_HeadAccelPSet = 2,
        ICPS_HeadGyroPSet = 3,
    };

    struct IMUCalData_t
    {
        XYZCoord_t AccelerometerGains;
        XYZCoord_t AccelerometerBias;
        IMUAccelerometerAxisCorrectionFactors_t AccelerometerRotationVals;

        XYZCoord_t GyroGains;
        XYZCoord_t GyroBias;
        IMUGyroAxisCorrectionFactors_t GyroRotationVals;

        //Clear sets the default values
        void Clear()
        {
            AccelerometerGains.Clear();
            AccelerometerBias.Clear();
            AccelerometerRotationVals.Clear();
            GyroGains.Clear();
            GyroBias.Clear();
            GyroRotationVals.Clear();

            AccelerometerGains.x = 1.0;
            AccelerometerGains.y = 1.0;
            AccelerometerGains.z = 1.0;

            GyroGains.x = 1.0;
            GyroGains.y = 1.0;
            GyroGains.z = 1.0;
        }
    };

    class IMUCalibration
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        int _calParamIdx = 0;

    public:
        IMUCalData_t IMUCalData;

        IMUCalibration();

        ~IMUCalibration(){};

        void Clear()
        {
            IMUCalData.Clear();
        }

        void ResetCalParamIdx()
        {
            _calParamIdx = 0;
        }

        //The Accelerometer Calibration Parameters are indexed
        //based on the IMU code:  Gx,Gy,Gz,Bx,By,Bz...
        IMUCalParameter_t GetNextAccelerometerCalParameter();

        //The Gyro Calibration Parameters are indexed
        //based on the IMU code:  Gx,Gy,Gz,Bx,By,Bz...
        IMUCalParameter_t GetNextGyroCalParameter();


        void ReadAccelerometerIMUCalInfo(ConfigData &calDataCfg);

        void ReadGyroIMUCalInfo(ConfigData &calDataCfg);

        //Read IMU Calibration from the given file.
        //The file name is the full path to the file.
        bool ReadIMUCalibrationFromIniFile(std::string filename);

    };

}
#endif //VIDERE_DEV_IMUCALIBRATION_H
