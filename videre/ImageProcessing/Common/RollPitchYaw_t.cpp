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

#include "RollPitchYaw_t.h"

namespace MathLibsNS
{

    const double RollPitchYaw_t::EqualEpslon = 1.0e-10;
    const double RollPitchYaw_t::STRPRECISION = 6;
    const double RollPitchYaw_t::STRPRECISIONCSV = 12;
    const double RollPitchYaw_t::PI = boost::math::constants::pi<double>();
    const double RollPitchYaw_t::TWOPI = 2.0 * boost::math::constants::pi<double>();
    const double RollPitchYaw_t::HALFPI = 0.5 * boost::math::constants::pi<double>();
    const double RollPitchYaw_t::RTOD = 180.0 / boost::math::constants::pi<double>();
    const double RollPitchYaw_t::DTOR = boost::math::constants::pi<double>() / 180.0;


    /**
     * Take the difference between two RollPitchYaw_t
     * This handles boundary conditions so if yaw1 = 10 degrees
     * and yaw2 = 350 degrees the results will be 20 degrees.
     * Note:  For delta Yaw, results between 180 and 360 are effectively
     * negative delta yaw.
     */
    RollPitchYaw_t RollPitchYaw_t::operator-(const RollPitchYaw_t &rpy) const
    {
        RollPitchYaw_t r;
        if(!IsRate)
        {
            double delRoll = _rollRad - rpy._rollRad;
            if (delRoll < -PI)
                delRoll += TWOPI;
            else if (delRoll >= PI)
                delRoll -= TWOPI;
            r._rollRad = delRoll;

            r.SetPitchRadians(_pitchRad - rpy._pitchRad);

            double delYaw = _yawRad - rpy._yawRad;
            if (delYaw < -PI)
                delYaw += TWOPI;
            else if (delYaw >= PI)
                delYaw -= TWOPI;
            r.SetYawRadians(delYaw);
        }
        else //This is rate in radians per second... no boundary roll-over issues.
        {
            r.SetRollRadians(_rollRad - rpy._rollRad);
            r.SetPitchRadians(_pitchRad - rpy._pitchRad);
            r.SetYawRadians(_yawRad - rpy._yawRad);
        }
        return r;
    }

    //Euclidean Distance between to vectors.
    //Sqroot of sum of square differences.
    //Used to test if two Roll Pitch Yaw vectors
    //are close to each other.
    double RollPitchYaw_t::EuclideanDistance(RollPitchYaw_t &rpy) const
    {
        RollPitchYaw_t delrpy;
        delrpy = *this - rpy;
        double dist = delrpy._rollRad * delrpy._rollRad;
        dist += delrpy._pitchRad * delrpy._pitchRad;
        double yaw = delrpy.YawPlusMinusRadians();
        dist += yaw * yaw;
        dist = sqrt(dist);
        return dist;
    }




}