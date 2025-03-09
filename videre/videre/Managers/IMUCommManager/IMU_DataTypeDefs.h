/* ****************************************************************
* Athr(s): Harry Direen PhD,
* DireenTech Inc.  (www.DireenTech.com)
* Date: Jan. 13, 2018
*
* NeuroGroove IMU Communications Interface
*******************************************************************/

#ifndef VIDERE_DEV_IMU_DATATYPEDEFS_H
#define VIDERE_DEV_IMU_DATATYPEDEFS_H

#include <math.h>

namespace IMU_SensorNS
{

#define IMU_MAXMESSAGESIZE 128

//IMU Sensor Info
//There a two sensors:
//		0:  Fixed Sensor on the uP Board
//		1:  Head Sensor... mounted on the user's head
#define TOTALNUMBEROFACCELGYROSENSORS 2
#define FIXEDSENSORIDX 0
#define HEADSENSORIDX 1


//Note:  the Imu_SensorId_e order is tied to the RS-232 output data format
//so changes here can impact client code.
    enum Imu_SensorId_e
    {
        IMUSensor_Fixed = FIXEDSENSORIDX,
        IMUSensor_Head = HEADSENSORIDX,
        IMUSensor_Both,
        IMUSensor_NA,
    };

//Note:  the dataType order is tied to the RS-232 output data format
//so changes here can impact client code.
    enum IMU_DataType_e
    {
        Accel_Raw,
        Accel_Ave,
        Accel_Statistics,
        Accel_Calibrated,
        Accel_Processed,
        Accel_WithOutGravity,
        GravityEstimate,
        Gyro_Raw,
        Gyro_Ave,
        Gyro_Statistics,
        Gyro_Calibrated,
        Gyro_Processed,
        Mag_Raw,
        Mag_Ave,
        Mag_Statistics,
        Mag_Calibrated,
        Mag_Processed,
        AccelGyro_Raw,
        AccelGyro_Ave,
        AccelGyro_Stdeviation,
        AccelGyro_Calibrated,
        AccelGyro_Processed,
        AccelGyro_WithOutGravity,
        SampleTimeStatistics,
        Position,
        Velocity,
        Orientation
    };

    inline bool IsDataTypeAccelerometer(IMU_DataType_e dt)
    {
        bool flg = dt == Accel_Raw;
        flg |= dt == Accel_Ave;
        flg |= dt == Accel_Statistics;
        flg |= dt == Accel_Calibrated;
        flg |= dt == Accel_Processed;
        flg |= dt == Accel_WithOutGravity;
        return flg;
    }

    inline bool IsDataTypeGyro(IMU_DataType_e dt)
    {
        bool flg = dt == Gyro_Raw;
        flg |= dt == Gyro_Ave;
        flg |= dt == Gyro_Statistics;
        flg |= dt == Gyro_Calibrated;
        flg |= dt == Gyro_Processed;
        return flg;
    }

    inline bool IsDataTypeMagnetometer(IMU_DataType_e dt)
    {
        bool flg = dt == Mag_Raw;
        flg |= dt == Mag_Ave;
        flg |= dt == Mag_Statistics;
        flg |= dt == Mag_Calibrated;
        flg |= dt == Mag_Processed;
        return flg;
    }

    inline bool IsDataTypeAccelGyro(IMU_DataType_e dt)
    {
        bool flg = dt == AccelGyro_Raw;
        flg |= dt == AccelGyro_Ave;
        flg |= dt == AccelGyro_Calibrated;
        flg |= dt == AccelGyro_Processed;
        return flg;
    }


//Note:  the Data_Format_t order is tied to the RS-232 output data format
//so changes here can impact client code.
    enum Data_Format_t
    {
        DFT_Cartesian,
        DFT_EulerAngles,
        DFT_Quaternion,
        DFT_AccelGyro,
        DFT_StatsAveStdDev,
        DFT_StatsSampleTimeAveStdDev,
    };

    typedef struct
    {
        IMU_DataType_e dataType;
        double x;
        double y;
        double z;
        double tstampSec;
    }Data_Cartesian_t;

    typedef struct
    {
        IMU_DataType_e dataType;
        double Ax;
        double Ay;
        double Az;
        double Gx;
        double Gy;
        double Gz;
        double tstampSec;		//Time in 1 microseconds steps???
    } Data_AccelGyro_t;

    typedef struct
    {
        IMU_DataType_e dataType;
        double Ave_x;
        double Ave_y;
        double Ave_z;
        double StdDev_x;
        double StdDev_y;
        double StdDev_z;
        double tstampSec;		//Time in 1 microseconds steps???
    } Data_StatsAveStdDev_t;

    typedef struct
    {
        IMU_DataType_e dataType;
        double Ave_dt;
        double StdDev_dt;
        double tstampSec;		//Time in 1 microseconds steps???
    } Data_StatsSampleTimeAveStdDev_t;


    typedef struct
    {
        IMU_DataType_e dataType;
        double theta;
        double phi;
        double psi;
        double tstampSec;
    } Data_EulerAngles_t;

    typedef struct
    {
        IMU_DataType_e dataType;
        double ux;
        double uy;
        double uz;
        double theta;
        double tstampSec;
    }Data_Quaterion_t;

    typedef union
    {
        Data_Cartesian_t C;
        Data_EulerAngles_t E;
        Data_Quaterion_t Q;
        Data_AccelGyro_t AG;
        Data_StatsAveStdDev_t StatsAveStd;
        Data_StatsSampleTimeAveStdDev_t StatsSmplTime;
    } DataGenericFormat_t;

    typedef struct
    {
        Imu_SensorId_e imuSource;    //Fixed, Head, Both, NA
        Data_Format_t dftype;
        DataGenericFormat_t data;
    }IMU_Data_t;

#define NUMBEROFImuProcessModeOfOperation 5
    typedef enum
    {
        ModeOfOperation_RawOutput,
        ModeOfOperation_SensorStatistics,
        ModeOfOperation_ComputeQuaternion,
        ModeOfOperation_Calibration,
        ModeOfOperation_StdOperation,

    }ImuProcessModeOfOperation_e;


    //Return true if data is valid and less than or equal to maxAGDataVals;
    //Return false if invalid data.
    bool isValidAccelGyroData(Data_AccelGyro_t &data,
                                Data_AccelGyro_t &maxAGDataVals);



}

#endif //VIDERE_DEV_IMU_DATATYPEDEFS_H
