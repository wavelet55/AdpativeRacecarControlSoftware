/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: March. 2016
 *
  *******************************************************************/

#ifndef VIDERE_DEV_QUATERNION_H
#define VIDERE_DEV_QUATERNION_H

#include <math.h>
#include <boost/math/constants/constants.hpp>
#include "XYZCoord_t.h"
#include "MathUtils.h"
#include <armadillo>


namespace MathLibsNS
{
    struct Quaternion_t
    {
        static const double PI;
        static const double TWOPI;
        static const double HALFPI;
        static const double RTOD;
        static const double DTOR;

    public:
        double qScale;
        XYZCoord_t qVec;

        //Creates an Identity Quaternion by default.
        Quaternion_t()
        {
            qScale = 1.0;
            qVec.Clear();
        }

        //Copy Constructor
        Quaternion_t(const Quaternion_t &q)
        {
            qScale = q.qScale;
            qVec = q.qVec;
        }

        void SetQuaternion(double s, double x, double y, double z)
        {
            qScale = s;
            qVec.SetXYZCoord(x, y, z);
        }

        void SetQuaternion(arma::mat &vec)
        {
            qScale = vec[0];
            qVec.SetXYZCoord(vec[1], vec[2], vec[3]);
        }

        void Clear()
        {
            qScale = 0;
            qVec.Clear();
        }

        //Make the Identity Quaternion...
        void MakeIdentity()
        {
            qScale = 1.0;
            qVec.Clear();
        }

        //Check to ensure all numbers are finite and not: NAN
        //Returns true if any value is a NAN or is not finite.
        bool CheckNAN();

        void MakeQuaternionFromVec(XYZCoord_t vec)
        {
            qScale = 0;
            qVec = vec;
        }

        void MakeQuaternionFromEulerAnglesRad(double Xrad, double Yrad, double Zrad);

        void MakeQuaternionFromGyroVec(XYZCoord_t &gyroRotVeldouble, double deltaTsec, double minGyroAngularVelocity = 1.0e-6);

        void MakeQuaternionFromGyroVec(double gyroX, double GyroY, double GyroZ, double deltaTsec, double minGyroAngularVelocity = 1.0e-6)
        {
            XYZCoord_t gVec(gyroX, GyroY, GyroZ);
            return MakeQuaternionFromGyroVec(gVec, deltaTsec, minGyroAngularVelocity);
        }

        void  MakeQuaternionFromRodriguesVector(XYZCoord_t &rVec);

        void  MakeQuaternionFromRodriguesVector(double x, double y, double z)
        {
            XYZCoord_t rVec(x, y, z);
            MakeQuaternionFromRodriguesVector(rVec);
        }

        XYZCoord_t MakeRodriguesVector();

        //Returns the Quaternion Norm
        double Norm()
        {
            double norm = qScale * qScale + qVec.MagnitudeSquared();
            return sqrt(norm);
        }

        void Normalize()
        {
            double norm = Norm();
            if(norm > 0)
            {
                double sf = 1.0 / norm;
                qScale = sf * qScale;
                qVec = qVec * sf;
            }
        }

        Quaternion_t getNormalizedQuaterion()
        {
            Quaternion_t qn(*this);
            qn.Normalize();
            return qn;
        }

        void Congugate()
        {
            qVec = qVec * -1.0;
        }

        Quaternion_t getCongugate()
        {
            Quaternion_t qc(*this);
            qc.qVec = qVec * -1.0;
            return qc;
        }

        Quaternion_t MultBy(Quaternion_t &q2);

        /**
         * Element-wise addition.
         */
        Quaternion_t operator+(const Quaternion_t &b) const
        {
            Quaternion_t r;
            r.qScale = qScale + b.qScale;
            r.qVec = qVec + b.qVec;
            return r;
        }

        /**
         * Element-wise subtraction.
         */
        Quaternion_t operator-(const Quaternion_t &b) const
        {
            Quaternion_t r;
            r.qScale = qScale - b.qScale;
            r.qVec = qVec - b.qVec;
            return r;
        }

        Quaternion_t operator*(const double c) const
        {
            Quaternion_t r;
            r.qScale = c * qScale;
            r.qVec = qVec * c;
            return r;
        }

        //Multiply this quaternion by the
        Quaternion_t operator*(const Quaternion_t &qRight) const;


        Quaternion_t operator/(const double c) const
        {
            Quaternion_t r;
            //Don't allow a 0 value for c to cause a fault.
            double sf = fabs(c) > 0 ? 1.0 / c : 1.0e12;
            r.qScale = sf * qScale;
            r.qVec = qVec * sf;
            return r;
        }

        //Generate a normalized Quaternion that will rotate vec1 to vec2.
        //Vectors 1 and 2 are expected to be normalized.
        static Quaternion_t Vec1ToVec2RotationQuaternion(XYZCoord_t &nvec1, const XYZCoord_t &nvec2);

        //Rotate the vector v by the normalized quaternion (this)
        XYZCoord_t rotateVecByQuaternion(XYZCoord_t &vec);

        void rotateVecByQuaternion(arma::mat &vecIn, arma::mat &vecOut);

        //The standard output is in radians. Set inDegrees=true for degrees output.
        //Assumes the quaternion is normalized
        XYZCoord_t toEulerAngles(bool inDegrees = false);

        //Turns quaternion into its matrix form. This is the form it takes if we
        //multiply q*p. In this case we have q on the left side, so it has the
        //form of a left matrix.
        //A 4x4 Armatillo matrix must be supplied.
        void toLeftMatrix(arma::mat &lMat);

        //Turns quaternion into its matrix form. This is the form it takes if we
        //multiply p*q. In this case we have q on the right side, so it has the
        //form of a right matrix.
        //A 4x4 Armatillo matrix must be supplied.
        void toRightMatrix(arma::mat &rMat);

        void toVector(arma::mat &vec)
        {
            vec(0) = qScale;
            vec(1) = qVec.x;
            vec(2) = qVec.y;
            vec(3) = qVec.z;
        }

    };

}
#endif //VIDERE_DEV_QUATERNION_H
