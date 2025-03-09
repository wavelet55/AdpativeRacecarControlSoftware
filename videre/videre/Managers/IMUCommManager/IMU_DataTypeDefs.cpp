/* ****************************************************************
* Athr(s): Harry Direen PhD,
* DireenTech Inc.  (www.DireenTech.com)
* Date: Jan. 13, 2018
*
* NeuroGroove IMU Communications Interface
*******************************************************************/

#include "IMU_DataTypeDefs.h"
#include <math.h>


namespace IMU_SensorNS
{


    //Return true if data is valid and less than or equal to maxAGDataVals;
    //Return false if invalid data.
    bool isValidAccelGyroData(Data_AccelGyro_t &data,
                               Data_AccelGyro_t &maxAGDataVals)
    {
        bool invalid = true;
        invalid &= isfinite(data.Ax) ? fabs(data.Ax) <= maxAGDataVals.Ax : false;
        invalid &= isfinite(data.Ay) ? fabs(data.Ay) <= maxAGDataVals.Ay : false;
        invalid &= isfinite(data.Az) ? fabs(data.Az) <= maxAGDataVals.Az : false;
        invalid &= isfinite(data.Gx) ? fabs(data.Gx) <= maxAGDataVals.Gx : false;
        invalid &= isfinite(data.Gy) ? fabs(data.Gy) <= maxAGDataVals.Gy : false;
        invalid &= isfinite(data.Gz) ? fabs(data.Gz) <= maxAGDataVals.Gz : false;
        invalid &= isfinite(data.tstampSec) ? fabs(data.tstampSec) <= maxAGDataVals.tstampSec : false;
        return invalid;
    }





}
