/* ****************************************************************
 * Image Processing Routine
 *
 * This routing was orginally developed by:
 *   Hyukseong Kwon, PhD
 *   June 2011 with modifications over time
 *
 * Modified for use with the Videre System by
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

#ifndef VIDERE_DEV_IMAGEPIXELLOCATIONTOREALWORLDLOCATION_H
#define VIDERE_DEV_IMAGEPIXELLOCATIONTOREALWORLDLOCATION_H

#include "global_defines.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "CameraCalibrationData.h"
#include "LatLonAltStruct.h"
#include "XYZCoord_t.h"
#include "RollPitchYaw_t.h"
#include "AzimuthElevation_t.h"
#include "config_data.h"
#include "../../Utils/logger.h"

using namespace videre;
using namespace GeoCoordinateSystemNS;
using namespace MathLibsNS;

namespace VidereImageprocessing
{

    //This object is used to transform Image Pixel locations
    //to ground X-Y position values along with Azimuth and Elevation
    //angles from the UAV location to the ground x-y position
    class ImagePixelLocationToRealWorldLocation
    {

    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;


        //Rotation Matricies for converting from Camera to real-world coordinates
        cv::Mat _tmp3x3Mtx;
        cv::Mat _rot3x3Mtx;

        //cv Matricies for Calculation
        cv::Mat _pixInp_2x1_Vec;
        cv::Mat _pixUnDist_2x1_Vec;

        cv::Mat _pixCoord_3x1_Vec;
        cv::Mat _uavCoord_3x1_Vec;


    public:
        ImageProcLibsNS::CameraCalibrationData CameraCalData;

        //The Vehicle X-Y Coordinate location
        //and the Altitude above ground level (all in meters)
        XYZCoord_t VehicleLoc;

        //If a TASE Gimble is being used... the camera Azimuth
        //and elevation is all that is required... so ignore the
        //camera rotation and translation compensation items
        //along with the UAV roll-pitch-yaw.
        bool IgnoreVehicleRollPitchYawCompensation = false;

        //A Flag to be used by the user to indicate whether or not
        //the calibration factors have been setup.
        bool CalibrationIsSetup = false;

        cv::Mat CameraTransVec;    //total camera translation 3x1 vector
        cv::Mat CameraRotMtx;      //total camera rotation 3x3 Matrix

    public:
        //Constructor
        ImagePixelLocationToRealWorldLocation();

        //Destructor
        ~ImagePixelLocationToRealWorldLocation();


        bool ReadCameraCalDataFromFile(const std::string &directory, const std::string &filename);

        bool ReadCameraCalDataFromFile(const std::string &fullDirFilename);

        void SetDefaultCameraCalData();

        //Undistort a point in the image plane
        //The method uses the Intrinsic calibration data and the Distortion coefficient
        //data to undistort the given image point and returns the undistored point in the
        //undistoredImgPt.
        void UndistorePointInImagePlane(const cv::Mat &inpImgPointM, cv::Mat &undistoredImgPointM);


        //The Pixel to Real-world location can be calculated at multiple image (pixel)
        //locations.  Use this method to set the current UAV/Vehicle location and
        //camera position before calling the CalculateRealWorldLocation method.
        //Camera Azimuth and Elevation are relative to the airframe, even if a TASE
        //gimble is being used.  Azimuth of 0 is straight ahead with positive angles
        //going to the right.  Elevation of 0 is stright ahead, -90 degrees is stright down
        //and +90 degrees is straith up.
        bool SetVehicleAndCameraLocation( XYZCoord_t &vehicleXYZLoc,
                                          RollPitchYaw_t &vehicleRollPitchYaw,
                                          AzimuthElevation_t &cameraAzimuthElev);


        //Calculate the the real-world or ground location of a given pixel location based upon the
        //UAV's current location and the camera's Azimuth and Elevation angles. These items are preset
        //with the SetVehicleAndCameraLocation(...) method.
        //Returns the Real-World X-Y coordinate location in xyRLoc along with the lenth of the
        //vector from the UAV to the xy location in the "z" parameter.
        //also returns the Azimuth and Elevation angles from the UAV to the target location.
        //Returns false if calculated ok... true if there was an error in the calculation process.
        bool CalculateRealWorldLocation(double pixX, double pixY,
                                        XYZCoord_t *xyRLoc,
                                        AzimuthElevation_t *azimuthElevFromUavToLoc);

    private:
        //Helper Methods

        void GenerateUavRollRotationMtx(double uavRollAngleRad);
        void GenerateUavPitchRotationMtx(double uavPitchAngleRad);
        void GenerateUavYawRotationMtx(double uavYawAngleRad);
        void GenerateCameraAzimuthRotationMtx(double cameraAzimuthAngleRad);
        void GenerateCameraElevationRotationMtx(double cameraElevationAngleRad);

    };

}


#endif //VIDERE_DEV_IMAGEPIXELLOCATIONTOREALWORLDLOCATION_H
