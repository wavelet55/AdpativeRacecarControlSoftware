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

#ifndef VIDERE_DEV_ROLLPITCHYAW_T_H
#define VIDERE_DEV_ROLLPITCHYAW_T_H

namespace MathLibsNS
{

    //Roll, Pitch Yaw for a aircraft;
    //This structure keeps these coordinats in radian format
    //and clamped to the standard ranges.  The class also handles
    //the math around cross-over points (180 or 360 degrees).
    //  Roll is around the x-axis or body of a aircraft.  Positive
    //  roll is right or starboard wing down (-180, 180]
    //  Pitch is around the y-axis or wings with postive pitch being up. [-90, 90]
    //  Yaw is aournd the z-axis (the postive z-axis is down).  Positive yaw
    //  is towards the right (starboard) wing.  Range: [0, 360.0).  Zero degress
    //  is often associated with true-north.
    struct RollPitchYaw_t
    {
    private:

        double _rollRad;
        double _pitchRad;
        double _yawRad;

        //Used to test equality between XYZCoord_t
        static const double EqualEpslon;
        static const double STRPRECISION;
        static const double STRPRECISIONCSV;
        static const double PI;
        static const double TWOPI;
        static const double HALFPI;
        static const double RTOD;
        static const double DTOR;

    public:
        //Fi the structure is used to store roll, pitch yaw rates
        //then this flage is true, otherwise it is false.
        bool IsRate;

        double RollRadians() const
        {
            return _rollRad;
        }

        double RollDegrees() const
        {
            return RTOD * _rollRad;
        }

        void SetRollRadians(double value)
        {
            if( !IsRate)
            {
                if (value <= -PI || value > PI)
                {
                    value = fmod(value, TWOPI);
                    if (value > PI)
                        value -= 2 * PI;
                    else if (value <= -PI)
                        value += 2 * PI;
                }
            }
            _rollRad = value;
        }

        void SetRollDegrees(double value)
        {
            SetRollRadians(DTOR * value);
        }

        double PitchRadians() const
        {
            return _pitchRad;
        }

        double PitchDegrees() const
        {
            return RTOD * _pitchRad;
        }

        void SetPitchRadians(double value)
        {
            if( !IsRate)
            {
                _pitchRad = value < -PI ? -PI : value > PI ? PI : value;
            }
            else
            {
                _pitchRad = value;
            }
        }

        void SetPitchDegrees(double value)
        {
            SetPitchRadians(DTOR * value);
        }

        //Yaw in the range: [0, 2*PI)
        double YawRadians() const
        {
            return _yawRad;
        }

        //Yaw in the range: [0, 360.0)
        double YawDegrees() const
        {
            return RTOD * _yawRad;
        }

        //Yaw in the range: (-PI, PI]
        double YawPlusMinusRadians()
        {
            if( _yawRad > PI)
                return _yawRad - TWOPI;
            else
                return _yawRad;
        }

        //Yaw in the range: (-180, +180.0)
        double YawPlusMinusDegrees() const
        {
            if( _yawRad > PI)
                return RTOD * (_yawRad - TWOPI);
            else
                return RTOD * _yawRad;
        }


        void SetYawRadians(double value)
        {
            if( !IsRate)
            {
                if (value < 0 || value >= TWOPI)
                {
                    value = fmod(value, TWOPI);
                    if (value < 0)
                        value += TWOPI;
                }
            }
            _yawRad = value;
        }

        void SetYawDegrees(double value)
        {
            SetYawRadians(DTOR * value);
        }

        RollPitchYaw_t()
        {
            _rollRad = 0;
            _pitchRad = 0;
            _yawRad = 0;
            IsRate = false;
        }


        RollPitchYaw_t(double roll, double pitch, double yaw, bool isRate, bool inDegrees = false)
        {
            IsRate = isRate;
           if(inDegrees)
           {
               SetRollDegrees(roll);
               SetPitchDegrees(pitch);
               SetYawDegrees(yaw);
           }
           else
           {
               SetRollRadians(roll);
               SetPitchRadians(pitch);
               SetYawRadians(yaw);
           }
        }

        void Clear()
        {
            _rollRad = 0;
            _pitchRad = 0;
            _yawRad = 0;
            IsRate = false;
        }

        /**
         * Element-wise addition, then adjusts to keep values in range.
         */
        RollPitchYaw_t operator+(const RollPitchYaw_t &rpy) const
        {
            RollPitchYaw_t r;
            r.SetRollRadians(_rollRad + rpy._rollRad);
            r.SetPitchRadians(_pitchRad + rpy._pitchRad);
            r.SetYawRadians(_yawRad + rpy._yawRad);
            return r;
        }

        /**
         * Take the difference between two RollPitchYaw_t
         * This handles boundary conditions so if yaw1 = 10 degrees
         * and yaw2 = 350 degrees the results will be 20 degrees.
         */
        RollPitchYaw_t operator-(const RollPitchYaw_t &rpy) const;


        /**
        * Multiple the roll, pitch and yaw rates times time to get
        * delta Roll, Pitch and Yaw angles.
        */
        RollPitchYaw_t RollPitchYawRatesXTimeSecs(double timeSec)
        {
            RollPitchYaw_t r;
            r.IsRate = false;
            r.SetRollRadians(_rollRad * timeSec);
            r.SetPitchRadians(_rollRad * timeSec);
            r.SetYawRadians(_rollRad * timeSec);
            return r;
        }

        //Euclidean Distance between to vectors.
        //Sqroot of sum of square differences.
        //Used to test if two Roll Pitch Yaw vectors
        //are close to each other.
        double EuclideanDistance(RollPitchYaw_t &rpy) const;

    };

}

#endif //VIDERE_DEV_ROLLPITCHYAW_T_H
