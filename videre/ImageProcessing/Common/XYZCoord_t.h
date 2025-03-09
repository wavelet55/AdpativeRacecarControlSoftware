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

#ifndef VIDERE_DEV_XYZCOORD_T_H
#define VIDERE_DEV_XYZCOORD_T_H

#include <string>
#include <sstream>
#include <iomanip>
#include <math.h>
#include <boost/math/constants/constants.hpp>
#include <armadillo>

namespace MathLibsNS
{

    /// <summary>
    /// X-Y-Z Coordinate structure.
    /// The z-coordinate is typically used for altitude,
    /// and is optional... it will be set to zero if not used
    /// which keeps it from affecting the X-Y (North/East)
    /// coordinates.
    /// This is a general purpose coordinate structure for doing
    /// math on these items.
    /// </summary>
    struct XYZCoord_t
    {
    public:
        double x;       //East / West
        double y;       //North / South
        double z;       //z is often the same as the altitude

        //Used to test equality between XYZCoord_t
        static const double EqualEpslon;

    private:
        static const double STRPRECISION;
        static const double STRPRECISIONCSV;


    public:
        XYZCoord_t()
        {
            x = 0.0;
            y = 0.0;
            z = 0.0;
        }

        XYZCoord_t(const XYZCoord_t &xyzVec)
        {
            x = xyzVec.x;
            y = xyzVec.y;
            z = xyzVec.z;
        }


        XYZCoord_t(double x_val, double y_val, double z_val = 0.0)
        {
            x = x_val;
            y = y_val;
            z = z_val;
        }

        XYZCoord_t(arma::mat &vec)
        {
            x = vec(0);
            y = vec(1);
            z = vec(2);
        }

        void SetXYZCoord(double x_val, double y_val, double z_val)
        {
            x = x_val;
            y = y_val;
            z = z_val;
        }

        void SetXYZCoord(arma::mat &vec)
        {
            x = vec(0);
            y = vec(1);
            z = vec(2);
        }

        void ToArmaVec3(arma::mat &vec)
        {
            vec(0) = x;
            vec(1) = y;
            vec(2) = z;
        }

        void Clear()
        {
            x = 0.0;
            y = 0.0;
            z = 0.0;
        }

        //Check to ensure all numbers are finite and not: NAN
        //Returns true if any value is a NAN or is not finite.
        bool CheckNAN();

        /**
         * Element-wise addition.
         */
        XYZCoord_t operator+(const XYZCoord_t &b) const
        {
            XYZCoord_t r;
            r.x = x + b.x;
            r.y = y + b.y;
            r.z = z + b.z;
            return r;
        }

        /**
         * Element-wise subtraction.
         */
        XYZCoord_t operator-(const XYZCoord_t &b) const
        {
            XYZCoord_t r;
            r.x = x - b.x;
            r.y = y - b.y;
            r.z = z - b.z;
            return r;
        }

        XYZCoord_t operator*(const double c) const
        {
            XYZCoord_t r;
            r.x = c * x;
            r.y = c * y;
            r.z = c * z;
            return r;
        }

        XYZCoord_t operator/(const double c) const;

        /// <summary>
        /// Magnitude of this.
        /// </summary>
        /// <returns></returns>
        double Magnitude();

        /// <summary>
        /// Magnitude Squared of this.
        /// </summary>
        /// <returns></returns>
        double MagnitudeSquared();

        /// <summary>
        /// City-Block magnitude.
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        double MagnitudeCityBlock();

        /// <summary>
        /// Distance between this xyCoord and "a".
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        double Distance(const XYZCoord_t &a);

        /// <summary>
        /// City-Block Distance between this xyCoord and "a".
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        double DistanceCityBlock(const XYZCoord_t &a);

        /// <summary>
        /// Calculate dot or inner product
        /// </summary>
        double InnerProduct(const XYZCoord_t &b);

        /// <summary>
        /// The Outer Product of the XY Vector with the a.XYVec.
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        double OuterProdXY(const XYZCoord_t &a);

        /// <summary>
        /// The cross product of this with vecR
        /// </summary>
        /// <param name="vecR"></param>
        /// <returns>this X vecR</returns>
        XYZCoord_t CrossProd(const XYZCoord_t &vecR);


        //Return a unit vector in the same direction
        //as this vector.  If the mag of the vector is near zero
        //return a zeroed vector.
        XYZCoord_t NormalizedVector();

        /// <summary>
        /// Rotate the X-Y vector (around the z-axis by theta)
        /// The z-axis is left un-changed.
        /// </summary>
        /// <param name="theta">angle to rotate</param>
        /// <param name="inDegrees">if true, theta is in degrees, otherwize theta is in radians</param>
        /// <returns></returns>
        XYZCoord_t RotateXYVec(double theta, bool inDegrees = false);

        /// <summary>
        /// Return an angle(radians) relative to the y-axis or North.
        /// </summary>
        /// <returns></returns>
        double HeadingDegrees();

        /// <summary>
        /// Return an angle(radians) relative to the y-axis or North.
        /// </summary>
        /// <returns></returns>
        double HeadingRadians();

        ///Set the X and Y values based upon the heading and magnitude;
        ///Zero degrees is North.  The z-axis is not changed.
        void SetHeadingDegMagnitudeXY(double headingDeg, double vecMag);

        ///Set the X and Y values based upon the heading and magnitude;
        ///Zero degrees is North.  The z-axis is not changed.
        void SetHeadingRadMagnitudeXY(double headingRad, double vecMag);



        std::string ToString();

        std::string ToCvsString();


    };

}

#endif //VIDERE_DEV_XYZCOORD_T_H
