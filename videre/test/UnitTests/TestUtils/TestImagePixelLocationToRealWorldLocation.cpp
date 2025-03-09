//
// Created by wavelet on 4/12/17.
//

#include "TestImagePixelLocationToRealWorldLocation.h"
#include "OpenCVMatUtils.h"
#include "XYZCoord_t.h"
#include <boost/math/constants/constants.hpp>
#include <math.h>

using namespace std;
using namespace videre;
using namespace VidereImageprocessing;
using namespace ImageProcLibsNS;
using namespace GeoCoordinateSystemNS;
using namespace MathLibsNS;


bool checkEqual(double a, double b, double eps = 1e-3)
{
    double delta = fabs(a - b);
    return delta < eps ? true : false;
}

double toRad(double angleDegrees)
{
    double PI = boost::math::constants::pi<double>();
    return (PI / 180.0) * angleDegrees;
}

double toDeg(double angleRad)
{
    double PI = boost::math::constants::pi<double>();
    return (180.0 / PI) * angleRad;
}

TEST_F(TestImagePixelLocationToRealWorldLocation, TestRotationMaticies)
{
    cv::Mat v1M(3, 1, CV_64F);
    cv::Mat v2M(3, 1, CV_64F);
    cv::Mat rotM(3, 3, CV_64F);
    cv::Mat rotMT(3, 3, CV_64F);
    XYZCoord_t v1(2.0, 5.0, 1.0);
    XYZCoord_t v2, v3;

    v1M.at<double>(0) = v1.x;
    v1M.at<double>(1) = v1.y;
    v1M.at<double>(2) = v1.z;

    Generate_Yaw_ZAxis_RotationMtx(toRad(90), rotM);
    v2M = rotM * v1M;

    v2.x = v2M.at<double>(0);
    v2.y = v2M.at<double>(1);
    v2.z = v2M.at<double>(2);

    //Expected output:
    v3 = XYZCoord_t(-5.0, 2.0, 1.0);

    double del = v2.Distance(v3);
    bool equal = checkEqual(del, 0);
    EXPECT_EQ(equal, true);

    Generate_Yaw_ZAxis_RotationMtx(toRad(180), rotM);
    v2M = rotM * v1M;

    v2.x = v2M.at<double>(0);
    v2.y = v2M.at<double>(1);
    v2.z = v2M.at<double>(2);

    //Expected output:
    v3 = XYZCoord_t(-2.0, -5.0, 1.0);

    del = v2.Distance(v3);
    equal = checkEqual(del, 0);
    EXPECT_EQ(equal, true);

    Generate_Roll_XAxis_RotationMtx(toRad(90), rotM);
    v2M = rotM * v1M;

    v2.x = v2M.at<double>(0);
    v2.y = v2M.at<double>(1);
    v2.z = v2M.at<double>(2);

    //Expected output:
    v3 = XYZCoord_t(2.0, -1.0, 5.0);

    del = v2.Distance(v3);
    equal = checkEqual(del, 0);
    EXPECT_EQ(equal, true);

    Generate_Pitch_YAxis_RotationMtx(toRad(90), rotM);
    v2M = rotM * v1M;

    v2.x = v2M.at<double>(0);
    v2.y = v2M.at<double>(1);
    v2.z = v2M.at<double>(2);

    //Expected output:
    v3 = XYZCoord_t(1.0, 5.0, -2.0);

    del = v2.Distance(v3);
    equal = checkEqual(del, 0);
    EXPECT_EQ(equal, true);

    //Combine 3 rotations
    Generate_Roll_XAxis_RotationMtx(toRad(90), rotMT);
    Generate_Pitch_YAxis_RotationMtx(toRad(90), rotM);
    rotMT = rotM * rotMT;
    Generate_Yaw_ZAxis_RotationMtx(toRad(90), rotM);
    rotMT = rotM * rotMT;

    v2M = rotMT * v1M;

    v2.x = v2M.at<double>(0);
    v2.y = v2M.at<double>(1);
    v2.z = v2M.at<double>(2);

    //Expected output:
    v3 = XYZCoord_t(1.0, 5.0, -2.0);

    del = v2.Distance(v3);
    equal = checkEqual(del, 0);
    EXPECT_EQ(equal, true);

    //Check Inverse of Rotation Matrix
    rotM = rotMT.inv();

    v1M.at<double>(0) = v3.x;
    v1M.at<double>(1) = v3.y;
    v1M.at<double>(2) = v3.z;

    v2M = rotM * v1M;

    v2.x = v2M.at<double>(0);
    v2.y = v2M.at<double>(1);
    v2.z = v2M.at<double>(2);

    del = v2.Distance(v1);
    equal = checkEqual(del, 0);
    EXPECT_EQ(equal, true);

    //Check Transpose of Rotation Matrix.
    rotM = rotMT.t();

    v2M = rotM * v1M;

    v2.x = v2M.at<double>(0);
    v2.y = v2M.at<double>(1);
    v2.z = v2M.at<double>(2);

    del = v2.Distance(v1);
    equal = checkEqual(del, 0);
    EXPECT_EQ(equal, true);

}

TEST_F(TestImagePixelLocationToRealWorldLocation, TestSetVehicleAndCameraLocation)
{
    ImagePixelLocationToRealWorldLocation xlator;
    XYZCoord_t vehicleXYZLoc(0, 0, 100.0);
    RollPitchYaw_t vehicleRollPitchYaw(0.0, 0.0, 0.0, false, true);
    AzimuthElevation_t cameraAzimuthElev(0.0, -90.0, true);
    cv::Mat v1M(3, 1, CV_64F);
    cv::Mat v2M(3, 1, CV_64F);
    XYZCoord_t v1, v2, v3;

    //The Default Camera Cal Rotation Matrix swaps the x and y axis, and mirrors the
    //y axis since larger y values were pointing backwards.
    xlator.SetDefaultCameraCalData();
    xlator.IgnoreVehicleRollPitchYawCompensation = true;

    xlator.SetVehicleAndCameraLocation(vehicleXYZLoc, vehicleRollPitchYaw, cameraAzimuthElev);

    //Test Image corrected pixel homogeneous vector
    v1 = XYZCoord_t(1.25, 0.75, 1.0);
    v1M.at<double>(0) = v1.x;
    v1M.at<double>(1) = v1.y;
    v1M.at<double>(2) = v1.z;

    v2M = xlator.CameraRotMtx * v1M;

    v2.x = v2M.at<double>(0);
    v2.y = v2M.at<double>(1);
    v2.z = v2M.at<double>(2);

    //Expected output:
    //ToDo:  This does not make sence to me...
    v3 = XYZCoord_t(-1, 1.25, -0.75);

    double del = v2.Distance(v3);
    bool equal = checkEqual(del, 0);
    EXPECT_EQ(equal, true);


    //EXPECT_EQ(cmp1, -1);

}
