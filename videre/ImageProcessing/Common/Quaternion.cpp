/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: March. 2016
 *
  *******************************************************************/

#include "Quaternion.h"
#include <math.h>

using namespace std;
using namespace arma;

namespace MathLibsNS
{

    const double Quaternion_t::PI = boost::math::constants::pi<double>();
    const double Quaternion_t::TWOPI = 2.0 * boost::math::constants::pi<double>();
    const double Quaternion_t::HALFPI = 0.5 * boost::math::constants::pi<double>();
    const double Quaternion_t::RTOD = 180.0 / boost::math::constants::pi<double>();
    const double Quaternion_t::DTOR = boost::math::constants::pi<double>() / 180.0;


    //Check to ensure all numbers are finite and not: NAN
    //Returns true if any value is a NAN or is not finite.
    bool Quaternion_t::CheckNAN()
    {
        bool invalidNumber = false;
        invalidNumber |= isnan(qScale) || !isfinite(qScale);
        invalidNumber |= isnan(qVec.x) || !isfinite(qVec.x);
        invalidNumber |= isnan(qVec.y) || !isfinite(qVec.y);
        invalidNumber |= isnan(qVec.z) || !isfinite(qVec.z);
        return invalidNumber;
    }

    //Multiply this quaternion by the
    Quaternion_t Quaternion_t::operator*(const Quaternion_t &qRight) const
    {
        Quaternion_t qr;
        qr.qScale = qScale * qRight.qScale - qVec.x * qRight.qVec.x - qVec.y * qRight.qVec.y - qVec.z * qRight.qVec.z;
        qr.qVec.x = qScale * qRight.qVec.x + qVec.x * qRight.qScale + qVec.y * qRight.qVec.z - qVec.z * qRight.qVec.y;
        qr.qVec.y = qScale * qRight.qVec.y - qVec.x * qRight.qVec.z + qVec.y * qRight.qScale + qVec.z * qRight.qVec.x;
        qr.qVec.z = qScale * qRight.qVec.z + qVec.x * qRight.qVec.y - qVec.y * qRight.qVec.x + qVec.z * qRight.qScale;
        return qr;
    }

    Quaternion_t Quaternion_t::MultBy(Quaternion_t &q2)
    {
        Quaternion_t qout;
        qout.qScale = qScale * q2.qScale - qVec.InnerProduct(q2.qVec);
        qout.qVec = q2.qVec * qScale + qVec * q2.qScale + qVec.CrossProd(q2.qVec);
        return qout;
    }

    void Quaternion_t::MakeQuaternionFromGyroVec(XYZCoord_t &gyroRotVel, double deltaTsec, double minGyroAngularVelocity)
    {
        double magRotVel = gyroRotVel.Magnitude();
        if(magRotVel > minGyroAngularVelocity)
        {
            double theta = 0.5 * magRotVel * deltaTsec;
            qScale = cos(theta);
            double sn = sin(theta);
            sn = sn / magRotVel;
            qVec = gyroRotVel * sn;
        }
        else
        {
            MakeIdentity();
        }
    }


    //Generate a normalized Quaternion that will rotate vec1 to vec2.
    //Vectors 1 and 2 are expected to be normalized.
    Quaternion_t Quaternion_t::Vec1ToVec2RotationQuaternion(XYZCoord_t &nvec1, const XYZCoord_t &nvec2)
    {
        Quaternion_t qr;
        qr.qScale = 1.0 + nvec1.InnerProduct(nvec2);
        qr.qVec = nvec1.CrossProd(nvec2);
        qr.Normalize();
        return qr;
    }

    //Rotate the vector v by the normalized quaternion (this)
    XYZCoord_t Quaternion_t::rotateVecByQuaternion(XYZCoord_t &vec)
    {
        XYZCoord_t r, vtmp, qxt;
        vtmp = qVec.CrossProd(vec) * 2.0;
        qxt = qVec.CrossProd(vtmp);
        vtmp = vtmp * qScale;
        r = vec + vtmp;
        r = r + qxt;
        return r;
    }

    void Quaternion_t::rotateVecByQuaternion(arma::mat &vecIn, arma::mat &vecOut)
    {
        XYZCoord_t vi(vecIn);
        XYZCoord_t vo;
        vo = rotateVecByQuaternion(vi);
        vo.ToArmaVec3(vecOut);
    }

    //The standard output is in radians. Set inDegrees=true for degrees output.
    //Assumes the quaternion is normalized
    XYZCoord_t Quaternion_t::toEulerAngles(bool inDegrees)
    {
        XYZCoord_t r;
        double sinr = 2.0 * (qScale * qVec.x + qVec.y * qVec.z);
        double cosr = 1.0 - 2.0 * (qVec.x * qVec.x + qVec.y * qVec.y);
        r.x = atan2(sinr, cosr);

        // pitch (y-axis rotation)
        double sinp = 2.0 * (qScale * qVec.y - qVec.z * qVec.x);
        if (fabs(sinp) >= 1.0)
            r.y = copysign(PI / 2, sinp); // use 90 degrees if out of range
        else
            r.y = asin(sinp);

        // yaw (z-axis rotation)
        double siny = +2.0 * (qScale * qVec.z + qVec.x * qVec.y);
        double cosy = +1.0 - 2.0 * (qVec.y * qVec.y + qVec.z * qVec.z);
        r.z = atan2(siny, cosy);

        if(inDegrees)
        {
            r = r * RTOD;
        }
        return r;
    }

    void Quaternion_t::MakeQuaternionFromEulerAnglesRad(double Xrad, double Yrad, double Zrad)
    {
        double cr = cos(Xrad * 0.5);
        double sr = sin(Xrad * 0.5);
        double cp = cos(Yrad * 0.5);
        double sp = sin(Yrad * 0.5);
        double cy = cos(Zrad * 0.5);
        double sy = sin(Zrad * 0.5);

        qScale = cy * cr * cp + sy * sr * sp;
        qVec.x = cy * sr * cp - sy * cr * sp;
        qVec.y = cy * cr * sp + sy * sr * cp;
        qVec.z = sy * cr * cp - cy * sr * sp;
    }

    void Quaternion_t::MakeQuaternionFromRodriguesVector(XYZCoord_t &rVec)
    {
        double mag = rVec.Magnitude();
        if(mag > 1e-6)
        {
            double halfTheta = 0.5 * mag;
            qScale = cos(halfTheta);
            double sf = sin(halfTheta) / mag;
            qVec = rVec * sf;
        }
        else
        {
            MakeIdentity();
        }
    }

    XYZCoord_t Quaternion_t::MakeRodriguesVector()
    {
        XYZCoord_t rVec;
        double wn = qScale < -1.0 ? -1.0 : qScale > 1.0 ? 1.0 : qScale;
        double thetaRad = 2.0 * acos(wn);
        rVec = qVec.NormalizedVector();
        return rVec * thetaRad;
    }

    //Turns quaternion into its matrix form. This is the form it takes if we
    //multiply q*p. In this case we have q on the left side, so it has the
    //form of a left matrix.
    //A 4x4 Armatillo matrix must be supplied.
    void Quaternion_t::toLeftMatrix(arma::mat &lMat)
    {
        if(lMat.n_rows == 4 && lMat.n_cols == 4)
        {
            lMat(0,0) = qScale;
            lMat(0,1) = -qVec.x;
            lMat(0,2) = -qVec.y;
            lMat(0,3) = -qVec.z;

            lMat(1,0) = qVec.x;
            lMat(1,1) = qScale;
            lMat(1,2) = -qVec.z;
            lMat(1,3) = qVec.y;

            lMat(2,0) = qVec.y;
            lMat(2,1) = qVec.z;
            lMat(2,2) = qScale;
            lMat(2,3) = -qVec.x;

            lMat(3,0) = qVec.z;
            lMat(3,1) = -qVec.y;
            lMat(3,2) = qVec.x;
            lMat(3,3) = qScale;
        }
    }


    //Turns quaternion into its matrix form. This is the form it takes if we
    //multiply p*q. In this case we have q on the right side, so it has the
    //form of a right matrix.
    //A 4x4 Armatillo matrix must be supplied.
    void Quaternion_t::toRightMatrix(arma::mat &rMat)
    {
        if(rMat.n_rows == 4 && rMat.n_cols == 4)
        {
            rMat(0,0) = qScale;
            rMat(0,1) = -qVec.x;
            rMat(0,2) = -qVec.y;
            rMat(0,3) = -qVec.z;

            rMat(1,0) = qVec.x;
            rMat(1,1) = qScale;
            rMat(1,2) = qVec.z;
            rMat(1,3) = -qVec.y;

            rMat(2,0) = qVec.y;
            rMat(2,1) = -qVec.z;
            rMat(2,2) = qScale;
            rMat(2,3) = qVec.x;

            rMat(3,0) = qVec.z;
            rMat(3,1) = qVec.y;
            rMat(3,2) = -qVec.x;
            rMat(3,3) = qScale;
        }
    }

}