//
// Created by wavelet on 8/17/16.
//

#include "TestLatLonAltCoord.h"
#include <memory>
#include <string>
#include <math.h>
#include <boost/math/constants/constants.hpp>
#include "GeoCoordinateSystem.h"
#include "LatLonAltStruct.h"
#include "AzimuthElevation_t.h"

using namespace std;
using namespace GeoCoordinateSystemNS;
using namespace MathLibsNS;

bool CompareDbls(double val1, double val2, double eps = 1e-6)
{
    double del = fabs(val1 - val2);
    bool equal = del < eps ? true : false;
    return equal;
}

TEST_F(TestLatLonAltCoord, TestLatLonAltGetSetMethods)
{
    double PI = boost::math::constants::pi<double>();
    double latDeg = 39.5;
    double lonDeg = -84.5 + 720.0;

    LatLonAltCoord_t lla1;
    lla1.SetLatitudeDegrees(latDeg);
    double lat1Deg = lla1.LatitudeDegrees();
    double lat1Rad = lla1.LatitudeRadians();
    double lat2Deg = (180.0 / PI) * lat1Rad;

    EXPECT_EQ(CompareDbls(lat1Deg, 39.5), true);
    EXPECT_EQ(CompareDbls(lat1Deg, lat2Deg), true);

    lla1.SetLongitudeDegrees(lonDeg);
    double lon1Deg = lla1.LongitudeDegrees();
    double lon1Rad = lla1.LongitudeRadians();
    double lon2Deg = (180.0 / PI) * lon1Rad;

    EXPECT_EQ(CompareDbls(lon1Deg, -84.5), true);
    EXPECT_EQ(CompareDbls(lon1Deg, lon2Deg), true);

    lon1Rad = (PI / 180.0) * lonDeg;
    lla1.SetLongitudeRadians(lon1Rad);
    lon1Deg = lla1.LongitudeDegrees();
    lon1Rad = lla1.LongitudeRadians();
    lon2Deg = (180.0 / PI) * lon1Rad;

    EXPECT_EQ(CompareDbls(lon1Deg, -84.5), true);
    EXPECT_EQ(CompareDbls(lon1Deg, lon2Deg), true);


    //EXPECT_NE(nullptr, msg1_ptr);
}