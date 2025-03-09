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

#include <string>
#include <sstream>
#include <iomanip>
#include <math.h>
#include <boost/math/constants/constants.hpp>

#ifndef VIDERE_DEV_AZIMUTHELEVATATION_T_H
#define VIDERE_DEV_AZIMUTHELEVATATION_T_H

namespace GeoCoordinateSystemNS
{

    //Camera Azimuth and Elevation angles;
    //This structure keeps these coordinates in radian format
    //and clamped to the standard ranges.  The class also handles
    //the math around cross-over points (180 or 360 degrees).
    struct AzimuthElevation_t
    {
    private:

        //Used to test equality between XYZCoord_t
        static const double EqualEpslon;
        static const double STRPRECISION;
        static const double STRPRECISIONCSV;
        static const double PI;
        static const double TWOPI;
        static const double HALFPI;
        static const double RTOD;
        static const double DTOR;

        /// <summary>
        /// Camera angle in radians with respect to the vehicle.
        /// Azimuth angle is left/right relative to a pilots view.
        /// Zero radians is striaght ahead.
        /// Positive angles are to the right, negative is to the left.
        /// </summary>
        double _azimuthAngleRad = 0;

        /// <summary>
        /// Camera angle in radians with respect to the vehicle.
        /// Elevation is up/down relative to a pilots view.
        /// Zero radians is straigh ahead.
        /// Positive angles are up, negative angles are down.
        /// -pi/2 is straight down relative to the vehicle.
        /// </summary>
        double _elevationAngleRad = 0;


    public:
        double AzimuthAngleRad()
        {
            return _azimuthAngleRad;
        }

        double AzimuthAngleDegrees()
        {
            return _azimuthAngleRad * RTOD;
        }

        void SetAzimuthAngleRad(double value)
        {
            if (value <= -PI || value > PI)
            {
                value = fmod(value, 2 * PI);
                if (value > PI)
                    value -= 2 * PI;
                else if (value <= -PI)
                    value += 2 * PI;
            }
            _azimuthAngleRad = value;
        }

        void SetAzimuthAngleDegrees(double value)
        {
            SetAzimuthAngleRad(value * DTOR);
        }

        double ElevationAngleRad()
        {
            return _elevationAngleRad;
        }

        double ElevationAngleDegrees()
        {
            return _elevationAngleRad * RTOD;
        }

        void SetElevationAngleRad(double value)
        {
            _elevationAngleRad = value < -1.0 * PI ? -1.0 * PI : value > PI ? PI : value;
        }

        void SetElevationAngleDegrees(double value)
        {
            SetElevationAngleRad(value * DTOR);
        }

        AzimuthElevation_t()
        {
            Clear();
        }

        AzimuthElevation_t(double azimuth, double elevation, bool inDegrees = false)
        {
           if(inDegrees)
           {
               SetAzimuthAngleDegrees(azimuth);
               SetElevationAngleDegrees(elevation);
           }
           else
           {
               SetAzimuthAngleRad(azimuth);
               SetElevationAngleRad(elevation);
           }
         }

        void Clear()
        {
            _azimuthAngleRad = 0;
            _elevationAngleRad = 0;
        }

        /**
         * Only Operates on Azimuth and elevation Angles.
         */
        AzimuthElevation_t operator+(const AzimuthElevation_t &aeAngles) const
        {
            AzimuthElevation_t r;
            r.SetAzimuthAngleRad(_azimuthAngleRad + aeAngles._azimuthAngleRad);
            r.SetElevationAngleRad(_elevationAngleRad + aeAngles._elevationAngleRad);
            return r;
        }

        /**
         * Take the difference between two AzimuthElevatation_t
         */
        AzimuthElevation_t operator-(const AzimuthElevation_t &rpy) const;



        //Euclidean Distance between to vectors.
        //Sqroot of sum of square differences.
        //Used to test if two Azimuth and Elevation vectors
        //are close to each other.
        double EuclideanDistance(AzimuthElevation_t &rpy) const;

    };

}

#endif //VIDERE_DEV_AzimuthElevatation_t_H


