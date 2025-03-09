/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date:  Jan. 2018
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#ifndef VIDERE_DEV_CAMERAORIENTATIONVALIDATION_H
#define VIDERE_DEV_CAMERAORIENTATIONVALIDATION_H

#include <limits>
#include <boost/math/constants/constants.hpp>
#include "../../Utils/config_data.h"
#include "VehicleInertialStatesMessage.h"
#include "CameraOrientationMessage.h"
#include "image_plus_metadata_message.h"

namespace TargetDetectorNS
{

    class CameraOrientationValidation
    {
        double RTOD = 180.0 / boost::math::constants::pi<double>();
        double DTOR = boost::math::constants::pi<double>() / 180.0;

    public:

        bool ValidateAltitudeAGL = false;
        double MinAltitudeAGL = 0;
        double MaxAltitudeAGL = 0;

        bool ValidateVehicleRollAngle = false;
        double MinRollAngleRadians = 0;
        double MaxRollAngleRadians = 0;

        bool ValidateVehiclePitchAngle = false;
        double MinPitchAngleRadians = 0;
        double MaxPitchAngleRadians = 0;

        bool ValidateCameraElevationAngle = false;
        double MinCameraElevationAngleRadians = 0;
        double MaxCameraElevationAngleRadians = 0;

        bool ValidateCameraAzimuthAngle = false;
        double MinCameraAzimuthAngleRadians = 0;
        double MaxCameraAzimuthAngleRadians = 0;

        double getMinRollAngleDegrees() {return RTOD * MinRollAngleRadians;}
        void setMinRollAngleDegrees(double valDeg)
        {
            valDeg = valDeg < -180.0 ? -180.0 : valDeg > 180.0 ? 180.0 : valDeg;
            MinRollAngleRadians = DTOR * valDeg;
        }

        double getMaxRollAngleDegrees() {return RTOD * MaxRollAngleRadians;}
        void setMaxRollAngleDegrees(double valDeg)
        {
            valDeg = valDeg < -180.0 ? -180.0 : valDeg > 180.0 ? 180.0 : valDeg;
            MaxRollAngleRadians = DTOR * valDeg;
        }

        double getMinPitchAngleDegrees() {return RTOD * MinPitchAngleRadians;}
        void setMinPitchAngleDegrees(double valDeg)
        {
            valDeg = valDeg < -180.0 ? -180.0 : valDeg > 180.0 ? 180.0 : valDeg;
            MinPitchAngleRadians = DTOR * valDeg;
        }

        double getMaxPitchAngleDegrees() {return RTOD * MaxPitchAngleRadians;}
        void setMaxPitchAngleDegrees(double valDeg)
        {
            valDeg = valDeg < -180.0 ? -180.0 : valDeg > 180.0 ? 180.0 : valDeg;
            MaxPitchAngleRadians = DTOR * valDeg;
        }

        double getMinCameraElevationAngleDegrees() {return RTOD * MinCameraElevationAngleRadians;}
        void setMinCameraElevationAngleDegrees(double valDeg)
        {
            valDeg = valDeg < -180.0 ? -180.0 : valDeg > 180.0 ? 180.0 : valDeg;
            MinCameraElevationAngleRadians = DTOR * valDeg;
        }

        double getMaxCameraElevationAngleDegrees() {return RTOD * MaxCameraElevationAngleRadians;}
        void setMaxCameraElevationAngleDegrees(double valDeg)
        {
            valDeg = valDeg < -180.0 ? -180.0 : valDeg > 180.0 ? 180.0 : valDeg;
            MaxCameraElevationAngleRadians = DTOR * valDeg;
        }

        double getMinCameraAzimuthAngleDegrees() {return RTOD * MinCameraAzimuthAngleRadians;}
        void setMinCameraAzimuthAngleDegrees(double valDeg)
        {
            valDeg = valDeg < -180.0 ? -180.0 : valDeg > 180.0 ? 180.0 : valDeg;
            MinCameraAzimuthAngleRadians = DTOR * valDeg;
        }

        double getMaxCameraAzimuthAngleDegrees() {return RTOD * MaxCameraAzimuthAngleRadians;}
        void setMaxCameraAzimuthAngleDegrees(double valDeg)
        {
            valDeg = valDeg < -360.0 ? -360.0 : valDeg > 360.0 ? 360.0 : valDeg;
            MaxCameraAzimuthAngleRadians = DTOR * valDeg;
        }

        CameraOrientationValidation()
        {
            Clear();
        }

        CameraOrientationValidation(const CameraOrientationValidation &cov)
        {
            *this = cov;
        }

        void Clear();

        void ReadValsFromConfigFile(std::shared_ptr<ConfigData> cfg);

        //Return true if ok, false if out of range.
        bool IsCameraOrientationInRange(ImagePlusMetadataMessage* imagePlusMetaData);

    };

}
#endif //VIDERE_DEV_CAMERAORIENTATIONVALIDATION_H
