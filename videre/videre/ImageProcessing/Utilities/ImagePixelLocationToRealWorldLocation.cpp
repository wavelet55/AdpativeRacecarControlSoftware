/* ****************************************************************
 * Image Processing Routine
 *
 * This routing was orginally developed by:
 *   Hyukseong Kwon, PhD
 *   June 2011 with modifications over time
 *
 * Nodified for use with the Videre System by
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: May 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#include <memory.h>
#include <math.h>
#include "ImagePixelLocationToRealWorldLocation.h"
#include <boost/math/constants/constants.hpp>
#include "../CameraCalibration/CameraCalReaderWriter.h"
#include "OpenCVMatUtils.h"
#include "FileUtils.h"
#include <stdio.h>
#include <opencv2/calib3d.hpp>

using namespace videre;
using namespace ImageProcLibsNS;
using namespace CameraCalibrationNS;
using namespace std;

namespace VidereImageprocessing
{


    ImagePixelLocationToRealWorldLocation::ImagePixelLocationToRealWorldLocation()
        : _pixInp_2x1_Vec(1, 2, CV_64FC2), _pixUnDist_2x1_Vec(1, 2, CV_64FC2),
          _rot3x3Mtx(3, 3, CV_64F), _tmp3x3Mtx(3, 3, CV_64F),
          CameraRotMtx(3, 3, CV_64F), CameraTransVec(3, 1, CV_64F),
          _pixCoord_3x1_Vec(3, 1, CV_64F), _uavCoord_3x1_Vec(3, 1, CV_64F)
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        IgnoreVehicleRollPitchYawCompensation = false;
        CalibrationIsSetup = false;
    }

    //Destructor
    ImagePixelLocationToRealWorldLocation::~ImagePixelLocationToRealWorldLocation()
    {
    }

    bool ImagePixelLocationToRealWorldLocation::ReadCameraCalDataFromFile(const std::string &directory, const std::string &filename)
    {
        bool error = false;
        try
        {
            boost::filesystem::path fullfilename(directory);
            fullfilename /= VidereFileUtils::AddOrReplaceFileExtention(filename, CAMERA_CAL_FILE_EXT);
            ReadCameraCalibrationFromIniFile(fullfilename.c_str(), CameraCalData);
            LOGINFO("Camera Cal Data Read from: " << filename);
            CalibrationIsSetup = true;
        }
        catch(exception e)
        {
            error = true;
            LOGWARN("Camera Cal Data Read Error, file: " << filename << " Ex:" << e.what());
            CalibrationIsSetup = false;
        }
        return error;
    }

    bool ImagePixelLocationToRealWorldLocation::ReadCameraCalDataFromFile(const std::string &fullDirFilename)
    {
        bool error = false;
        try
        {
            ReadCameraCalibrationFromIniFile(fullDirFilename, CameraCalData);
            CalibrationIsSetup = true;
        }
        catch(exception e)
        {
            error = true;
            CalibrationIsSetup = false;
        }
        return error;
    }

    void ImagePixelLocationToRealWorldLocation::SetDefaultCameraCalData()
    {
        CameraCalData.ClearAll();
        CameraCalData.SetDefaults();
        CalibrationIsSetup = true;
    }


    //The Pixel to Real-world location can be calculated at multiple image (pixel)
    //locations.  Use this method to set the current UAV/Vehicle location and
    //camera position before calling the CalculateRealWorldLocation method.
    //This method generates a combined Rotation Matrix that is used by the
    //CalculateRealWorldLocation method._
    bool ImagePixelLocationToRealWorldLocation::SetVehicleAndCameraLocation( XYZCoord_t &vehicleXYZLoc,
                                                                             RollPitchYaw_t &vehicleRollPitchYaw,
                                                                             AzimuthElevation_t &cameraAzimuthElev)
    {
        bool error = false;
        VehicleLoc = vehicleXYZLoc;
        double halfpi = 0.5 * boost::math::constants::pi<double>();
        try
        {
            //_cameraToRealWorldMtx = R_azimuth * R_elevation
            //Since the camera is assumed to be pointing straight down (normal mounting,
            //and ElevationAngle = -90 degrees), we will add 90 degrees to the elevation
            //angle to make things work out right.  The cvRotationCalM matrix is based
            //upon the standard camera mounting pointed stright down.  Another approach
            //could be to modify the cvRotationCalM to make things work out right.
            double elevationAngle = cameraAzimuthElev.ElevationAngleRad() + halfpi;
            Generate_Pitch_YAxis_RotationMtx(elevationAngle, _tmp3x3Mtx);
            Generate_Yaw_ZAxis_RotationMtx(cameraAzimuthElev.AzimuthAngleRad(), _rot3x3Mtx);
            _rot3x3Mtx = _rot3x3Mtx * _tmp3x3Mtx;

            if (!IgnoreVehicleRollPitchYawCompensation)
            {
                //_cameraToRealWorldMtx = R_yaw * R_pitch * R_roll * R_azimuth * R_elevation
                Generate_Roll_XAxis_RotationMtx(vehicleRollPitchYaw.RollRadians(), _tmp3x3Mtx);
                _rot3x3Mtx = _tmp3x3Mtx * _rot3x3Mtx;

                Generate_Pitch_YAxis_RotationMtx(vehicleRollPitchYaw.PitchRadians(), _tmp3x3Mtx);
                _rot3x3Mtx = _tmp3x3Mtx * _rot3x3Mtx;

                Generate_Yaw_ZAxis_RotationMtx(vehicleRollPitchYaw.YawRadians(), _tmp3x3Mtx);
                _rot3x3Mtx = _tmp3x3Mtx * _rot3x3Mtx;
            }
            //Now apply the camera cal (camera physical mounting correction).
            CameraRotMtx = _rot3x3Mtx * CameraCalData.cvRotationCalM;
            CameraTransVec = _rot3x3Mtx * CameraCalData.cvTranslationCalM;
        }
        catch(std::exception &e)
        {
            LOGERROR("SetVehicleAndCameraLocation: Exception" << e.what());
            error = true;
        }
        return error;
    }

    //Undistort a point in the image plane
    //The method uses the Intrinsic calibration data and the Distortion coefficient
    //data to undistort the given image point and returns the undistored point in the
    //undistoredImgPt.
    void ImagePixelLocationToRealWorldLocation::UndistorePointInImagePlane(const cv::Mat &inpImgPointM,
                                                                           cv::Mat &undistoredImgPointM)
    {
        cv::undistortPoints(inpImgPointM, undistoredImgPointM,
                          CameraCalData.cvIntrinsicCalM, CameraCalData.cvDistortionCoeff);
        //Scale Output by the Calibration Scale Factor
        undistoredImgPointM.at<double>(0)  = undistoredImgPointM.at<double>(0) * CameraCalData.GetCalibrationScaleFactorInverse();
        undistoredImgPointM.at<double>(1) = undistoredImgPointM.at<double>(1) * CameraCalData.GetCalibrationScaleFactorInverse();
    }

    //Calculate the the real-world or ground location of a given pixel location based upon the
    //UAV's current location and the camera's Azimuth and Elevation angles
    //Output Location:
    //  x-pos
    //  y-pos
    //  z = length of the vector from the UAV to the target x-y location.
    //  azimuth to target from uav
    //  elevation to target from uav
    bool ImagePixelLocationToRealWorldLocation::CalculateRealWorldLocation(double pixX, double pixY,
                                                                           XYZCoord_t *xyRLoc,
                                                                           AzimuthElevation_t *azimuthElevFromUavToLoc)
    {
        bool error = false;
        double pi = boost::math::constants::pi<double>();
        try
        {
            _pixInp_2x1_Vec.at<double>(0) = pixX;
            _pixInp_2x1_Vec.at<double>(1) = pixY;

            //Undistore the pixel location.
            //Generates a normalized pixel coordinates (focal-lengths = 1.0).
            UndistorePointInImagePlane(_pixInp_2x1_Vec, _pixUnDist_2x1_Vec);

            // Convert into the CAMERA coordinate frame
            _pixCoord_3x1_Vec.at<double>(0) = _pixUnDist_2x1_Vec.at<double>(0);
            _pixCoord_3x1_Vec.at<double>(1) = _pixUnDist_2x1_Vec.at<double>(1);
            _pixCoord_3x1_Vec.at<double>(2) = 1.0;

            _uavCoord_3x1_Vec = CameraRotMtx * _pixCoord_3x1_Vec + CameraTransVec;
            //cv::multiply(_cameraRotMtx, _pixCoord_3x1_Vec, _uavCoord_3x1_Vec);
            //cv::add(_uavCoord_3x1_Vec,  _cameraTransVec, _uavCoord_3x1_Vec);

            // Estimate azimuth and elevation angles to the target
            double azimuth = atan2(_uavCoord_3x1_Vec.at<double>(1), _uavCoord_3x1_Vec.at<double>(0));
            double elevation = fabs(atan2(_uavCoord_3x1_Vec.at<double>(2), sqrt(std::pow(_uavCoord_3x1_Vec.at<double>(0), 2.0)
                                                                         + pow(_uavCoord_3x1_Vec.at<double>(1), 2.0))));

            // Estimate the global(or local) location from azimuth and elevation angles
            //Clamp theta so we cannot point straight at the horizon
            elevation =
                    elevation < 0.01 * pi ? 0.01 * pi : elevation > (1.0 - 0.01) * pi ? (1.0 - 0.01) * pi : elevation;
            double theta = 0.5 * pi - elevation;
            double ground_dist = VehicleLoc.z * tan(theta);
            double R = sqrt(VehicleLoc.z * VehicleLoc.z + ground_dist * ground_dist);
            double x_uav = ground_dist * sin(azimuth);
            double y_uav = ground_dist * cos(azimuth);

            xyRLoc->x = VehicleLoc.x + x_uav;
            xyRLoc->y = VehicleLoc.y + y_uav;
            xyRLoc->z = R;
            azimuthElevFromUavToLoc->SetAzimuthAngleRad(azimuth);
            azimuthElevFromUavToLoc->SetElevationAngleRad(elevation);
        }
        catch (std::exception &e)
        {
            LOGERROR("CalculateRealWorldLocation Exception: " << e.what());
            error = true;
        }

        return error;
    }


}
