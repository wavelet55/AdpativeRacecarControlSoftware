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

#include "XYZCoord_t.h"
#include <math.h>

using namespace std;

namespace MathLibsNS
{

    const double XYZCoord_t::EqualEpslon = 1.0e-10;
    const double XYZCoord_t::STRPRECISION = 6;
    const double XYZCoord_t::STRPRECISIONCSV = 12;

    XYZCoord_t XYZCoord_t::operator/(const double c) const
    {
        XYZCoord_t r;
        //Handle a divide by zero is some reasonable fashion that
        //does not throw exceptions.
        double a = 1.0;
        if( fabs(c) > 0 )
            a = 1.0 / c;
        else if( c >= 0)
            a = 1.0 / EqualEpslon;
        else
            a = -1.0 / EqualEpslon;

        r.x = x * a;
        r.y = y * a;
        r.z = z * a;
        return r;
    }

    //Check to ensure all numbers are finite and not: NAN
    //Returns true if any value is a NAN or is not finite.
    bool XYZCoord_t::CheckNAN()
    {
        bool invalidNumber = false;
        invalidNumber |= isnan(x) || !isfinite(x);
        invalidNumber |= isnan(y) || !isfinite(y);
        invalidNumber |= isnan(z) || !isfinite(z);
        return invalidNumber;
    }


    /// <summary>
    /// Magnitude of this.
    /// </summary>
    /// <returns></returns>
    double XYZCoord_t::Magnitude()
    {
        return sqrt(x * x + y * y + z * z);
    }

    /// <summary>
    /// Magnitude Squared of this.
    /// </summary>
    /// <returns></returns>
    double XYZCoord_t::MagnitudeSquared()
    {
        return x * x + y * y + z * z;
    }


    /// <summary>
    /// City-Block magnitude.
    /// </summary>
    /// <param name="a"></param>
    /// <returns></returns>
    double XYZCoord_t::MagnitudeCityBlock()
    {
        double dx = fabs(x);
        double dy = fabs(y);
        double dz = fabs(z);
        double mxy = dx > dy ? dx : dy;
        return mxy > dz ? mxy : dz;
    }

    /// <summary>
    /// Distance between this xyCoord and "a".
    /// </summary>
    /// <param name="a"></param>
    /// <returns></returns>
    double XYZCoord_t::Distance(const XYZCoord_t &a)
    {
        double dx = x - a.x;
        double dy = y - a.y;
        double dz = z - a.z;
        return sqrt(dx * dx + dy * dy + dz * dz);
    }

    /// <summary>
    /// City-Block Distance between this xyCoord and "a".
    /// </summary>
    /// <param name="a"></param>
    /// <returns></returns>
    double XYZCoord_t::DistanceCityBlock(const XYZCoord_t &a)
    {
        double dx = fabs(x - a.x);
        double dy = fabs(y - a.y);
        double dz = fabs(z - a.z);
        double mxy = dx > dy ? dx : dy;
        return mxy > dz ? mxy : dz;
    }

    /// <summary>
    /// Calculate dot or inner product
    /// </summary>
    double XYZCoord_t::InnerProduct(const XYZCoord_t &b)
    {
        double c= x * b.x + y * b.y + z * b.z;
        return c;
    }

    /// <summary>
    /// The Outer Product of the XY Vector with the a.XYVec.
    /// </summary>
    /// <param name="a"></param>
    /// <returns></returns>
    double XYZCoord_t::OuterProdXY(const XYZCoord_t &a)
    {
        return x * a.y - y * a.x;
    }

    /// <summary>
    /// The cross product of this with vecR
    /// </summary>
    /// <param name="vecR"></param>
    /// <returns>this X vecR</returns>
    XYZCoord_t XYZCoord_t::CrossProd(const XYZCoord_t &vecR)
    {
        XYZCoord_t r;
        r.x = y * vecR.z - z * vecR.y;
        r.y = z * vecR.x - x * vecR.z;
        r.z = x * vecR.y - y * vecR.x;
        return r;
    }

    //Return a unit vector in the same direction
    //as this vector.  If the mag of the vector is near zero
    //return a zeroed vector.
    XYZCoord_t XYZCoord_t::NormalizedVector()
    {
        double d = Magnitude();
        XYZCoord_t r(*this);
        if (d > EqualEpslon)
        {
            d = 1.0 / d;
            r.x *= d;
            r.y *= d;
            r.z *= d;
        }
        else
        {
            r.Clear();
        }
        return r;
    }

    /// <summary>
    /// Rotate the X-Y vector (around the z-axis by theta)
    /// The z-axis is left un-changed.
    /// </summary>
    /// <param name="theta">angle to rotate</param>
    /// <param name="inDegrees">if true, theta is in degrees, otherwize theta is in radians</param>
    /// <returns></returns>
    XYZCoord_t XYZCoord_t::RotateXYVec(double theta, bool inDegrees)
    {
        XYZCoord_t rVec;
        if (inDegrees)
        {
            theta = (boost::math::constants::pi<double>() / 180.0) * theta;
        }
        double a = cos(theta);
        double b = sin(theta);
        rVec.x = a * x - b * y;
        rVec.y = b * x + a * y;
        return rVec;
    }

    /// <summary>
    /// Return an angle(radians) relative to the y-axis or North.
    /// Zero degrees is North
    /// </summary>
    /// <returns></returns>
    double XYZCoord_t::HeadingDegrees()
    {
        return (180.0 / boost::math::constants::pi<double>()) * atan2(x, y);
    }

    /// <summary>
    /// Return an angle(radians) relative to the y-axis or North.
    /// Zero degrees is North
    /// </summary>
    /// <returns></returns>
    double XYZCoord_t::HeadingRadians()
    {
        return atan2(x, y);
    }

    ///Set the X and Y values based upon the heading and magnitude;
    ///Zero degrees is North.  The z-axis is not changed.
    void XYZCoord_t::SetHeadingRadMagnitudeXY(double headingRad, double mag)
    {
        x = mag * sin(headingRad);
        y = mag * cos(headingRad);
    }


    ///Set the X and Y values based upon the heading and magnitude;
    ///Zero degrees is North.  The z-axis is not changed.
    void XYZCoord_t::SetHeadingDegMagnitudeXY(double headingDeg, double mag)
    {
        double rad = (boost::math::constants::pi<double>() / 180.0) * headingDeg;
        x = mag * sin(rad);
        y = mag * cos(rad);
    }


    std::string XYZCoord_t::ToString()
    {
        std::ostringstream buff;
        buff << "X=";
        buff << std::scientific << std::setprecision(STRPRECISION) << x;
        buff << ", Y=";
        buff << std::scientific << std::setprecision(STRPRECISION) << y;
        buff << ", Z=";
        buff << std::scientific << std::setprecision(STRPRECISION) << z;
        return buff.str();
    }

    std::string XYZCoord_t::ToCvsString()
    {
        std::ostringstream buff;
        buff << std::scientific << std::setprecision(STRPRECISIONCSV) << x;
        buff << ", ";
        buff << std::scientific << std::setprecision(STRPRECISIONCSV) << y;
        buff << ", ";
        buff << std::scientific << std::setprecision(STRPRECISIONCSV) << z;
        return buff.str();
    }

}

