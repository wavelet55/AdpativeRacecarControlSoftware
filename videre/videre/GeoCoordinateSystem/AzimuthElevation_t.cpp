/* ****************************************************************
* Athr(s): Harry Direen PhD, Randy Direen Phd.
* DireenTech Inc.  (www.DireenTech.com)
* Date: March. 2016
*
* Academy Center for UAS Research
* Department of Electrical and Computer Engineering
        * HQ USAFA/DFEC
* 2354 Fairchild Drive
* USAF Academy, CO 80840-6236
*
*******************************************************************/

#include "AzimuthElevation_t.h"
#include <math.h>
#include <boost/math/constants/constants.hpp>

namespace GeoCoordinateSystemNS
{

    const double AzimuthElevation_t::EqualEpslon = 1.0e-10;
    const double AzimuthElevation_t::STRPRECISION = 6;
    const double AzimuthElevation_t::STRPRECISIONCSV = 12;
    const double AzimuthElevation_t::PI = boost::math::constants::pi<double>();
    const double AzimuthElevation_t::TWOPI = 2.0 * boost::math::constants::pi<double>();
    const double AzimuthElevation_t::HALFPI = 0.5 * boost::math::constants::pi<double>();
    const double AzimuthElevation_t::RTOD = 180.0 / boost::math::constants::pi<double>();
    const double AzimuthElevation_t::DTOR = boost::math::constants::pi<double>() / 180.0;


    /**
      * Take the difference between two AzimuthElevatation_t
      */
    AzimuthElevation_t AzimuthElevation_t::operator-(const AzimuthElevation_t &rpy) const
    {
        AzimuthElevation_t r;
        double delAzimuth = _azimuthAngleRad - rpy._azimuthAngleRad;
        if (delAzimuth < -PI)
            delAzimuth += TWOPI;
        else if (delAzimuth >= PI)
            delAzimuth -= TWOPI;
        r._azimuthAngleRad = delAzimuth;

        r.SetElevationAngleRad(_elevationAngleRad - rpy._elevationAngleRad);
        return r;
    }


    //Euclidean Distance between to vectors.
    //Sqroot of sum of square differences.
    //Used to test if two Azimuth and Elevation vectors
    //are close to each other.
    double AzimuthElevation_t::EuclideanDistance(AzimuthElevation_t &aeAngles) const
    {
        AzimuthElevation_t delae;
        delae = *this - aeAngles;
        double dist = delae._azimuthAngleRad * delae._azimuthAngleRad;
        dist += delae._elevationAngleRad * delae._elevationAngleRad;
        dist = sqrt(dist);
        return dist;
    }

}