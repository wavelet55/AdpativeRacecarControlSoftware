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


#include "CameraOrientationValidation.h"

namespace TargetDetectorNS
{

    void CameraOrientationValidation::Clear()
    {
        ValidateAltitudeAGL = false;
        MinAltitudeAGL = 0;
        MaxAltitudeAGL = 0;

        ValidateVehicleRollAngle = false;
        MinRollAngleRadians = 0;
        MaxRollAngleRadians = 0;

        ValidateVehiclePitchAngle = false;
        MinPitchAngleRadians = 0;
        MaxPitchAngleRadians = 0;

        ValidateCameraElevationAngle = false;
        MinCameraElevationAngleRadians = 0;
        MaxCameraElevationAngleRadians = 0;

        ValidateCameraAzimuthAngle = false;
        MinCameraAzimuthAngleRadians = 0;
        MaxCameraAzimuthAngleRadians = 0;
    }

    void CameraOrientationValidation::ReadValsFromConfigFile(std::shared_ptr<ConfigData> cfg)
    {
        double val;
        ValidateAltitudeAGL = cfg->GetConfigBoolValue("CameraOrientationValidation.ValidateAltitudeAGL", false);
        MinAltitudeAGL = cfg->GetConfigDoubleValue("CameraOrientationValidation.MinAltitudeAGL", 100);
        MaxAltitudeAGL = cfg->GetConfigDoubleValue("CameraOrientationValidation.MaxAltitudeAGL", 500);

        ValidateVehicleRollAngle = cfg->GetConfigBoolValue("CameraOrientationValidation.ValidateVehicleRollAngle", false);
        val = cfg->GetConfigDoubleValue("CameraOrientationValidation.MinRollAngleDegrees", -60);
        setMinRollAngleDegrees(val);
        val = cfg->GetConfigDoubleValue("CameraOrientationValidation.MaxRollAngleDegrees", 60);
        setMaxRollAngleDegrees(val);

        ValidateVehiclePitchAngle = cfg->GetConfigBoolValue("CameraOrientationValidation.ValidateVehiclePitchAngle", false);
        val = cfg->GetConfigDoubleValue("CameraOrientationValidation.MinPitchAngleDegrees", 60);
        setMinPitchAngleDegrees(val);
        val = cfg->GetConfigDoubleValue("CameraOrientationValidation.MaxPitchlAngleDegrees", 60);
        setMaxPitchAngleDegrees(val);

        ValidateCameraElevationAngle = cfg->GetConfigBoolValue("CameraOrientationValidation.ValidateCameraElevationAngle", false);
        val = cfg->GetConfigDoubleValue("CameraOrientationValidation.MinCameraElevationAngleDegrees", -90);
        setMinCameraElevationAngleDegrees(val);
        val = cfg->GetConfigDoubleValue("CameraOrientationValidation.MaxCameraElevationAngleDegrees", -30);
        setMaxCameraElevationAngleDegrees(val);

        ValidateCameraAzimuthAngle = cfg->GetConfigBoolValue("CameraOrientationValidation.ValidateCameraAzimuthAngle", false);
        val = cfg->GetConfigDoubleValue("CameraOrientationValidation.MinCameraAzimuthAngleDegrees", -360);
        setMinCameraAzimuthAngleDegrees(val);
        val = cfg->GetConfigDoubleValue("CameraOrientationValidation.MaxCameraAzimuthAngleDegrees", 360);
        setMaxCameraAzimuthAngleDegrees(val);

    }

    //Return true if ok, false if out of range.
    bool CameraOrientationValidation::IsCameraOrientationInRange(ImagePlusMetadataMessage* imagePlusMetaData)
    {
        bool ok = true;
        if(ValidateAltitudeAGL)
        {
            if(imagePlusMetaData->VehicleInertialStates.HeightAGL < MinAltitudeAGL
               || imagePlusMetaData->VehicleInertialStates.HeightAGL > MaxAltitudeAGL)
            {
                return false;
            }
        }

        if(ValidateVehiclePitchAngle)
        {
            if (imagePlusMetaData->VehicleInertialStates.RollPitchYaw.PitchRadians() < MinPitchAngleRadians
                || imagePlusMetaData->VehicleInertialStates.RollPitchYaw.PitchRadians() > MaxPitchAngleRadians)
            {
                return false;
            }
        }

        if(ValidateVehicleRollAngle)
        {
            if (imagePlusMetaData->VehicleInertialStates.RollPitchYaw.RollRadians() < MinRollAngleRadians
                || imagePlusMetaData->VehicleInertialStates.RollPitchYaw.RollRadians() > MaxRollAngleRadians)
            {
                return false;
            }
        }

        if(ValidateCameraElevationAngle)
        {
            if (imagePlusMetaData->CameraOrientation.CameraAzimuthElevationAngles.ElevationAngleRad() < MinCameraElevationAngleRadians
                 ||   imagePlusMetaData->CameraOrientation.CameraAzimuthElevationAngles.ElevationAngleRad() > MaxCameraElevationAngleRadians)
            {
                return false;
            }
        }

        if(ValidateCameraAzimuthAngle)
        {
            if (imagePlusMetaData->CameraOrientation.CameraAzimuthElevationAngles.AzimuthAngleRad() < MinCameraAzimuthAngleRadians
                ||   imagePlusMetaData->CameraOrientation.CameraAzimuthElevationAngles.AzimuthAngleRad() > MaxCameraAzimuthAngleRadians)
            {
                return false;
            }
        }

        return ok;
    }

}