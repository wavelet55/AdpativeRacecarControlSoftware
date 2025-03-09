/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/

#ifndef VIDERE_DEV_IMU_RXMESSAGEFORMATTER_H
#define VIDERE_DEV_IMU_RXMESSAGEFORMATTER_H

#include "IMU_DataTypeDefs.h"

namespace IMU_SensorNS
{

    class IMU_TxMessageFormatter
    {



    public:
        IMU_TxMessageFormatter() {}

        int buildCmdWithSensorID(char *cmdBuf, const char *cmd, Imu_SensorId_e sID);
        int buildCmd(char *cmdBuf, const char *cmd);

        int addOnOffStr(char *cmdBuf, int offset, bool onOff, bool termCmd = true);

        //Note... most the commands support getting the IMU setting
        //If the getSetting flag is true, then a command for retrieving the
        //setting will be sent out.  The command does not wait for the response.
        //The Receive process will receive the commands response and handle
        //that message.  Any variable with the command will be ignored when
        //getSetting = true;

        //The commands return the string size of the command.
        //Each function requires a command buffer input with minimum size of: IMU_MAXMESSAGESIZE

        //enable or disable the Accelerometer at the sensor
        int AccelGyroEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        //Set the Accelerometer and Gyro Data Rate Diveder
        //Sample rate =: 1.125 kHz / (1 + val)
        int AccelGyroDataRateDivider(char *cmdBuf, Imu_SensorId_e sID, int val, bool getSetting = false);


        int AccelRawOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        int AccelProcessedOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        int AccelAveOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        int AccelStdDevOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        int AccelSetFullScale(char *cmdBuf, Imu_SensorId_e sID, int val, bool getSetting = false);

        int AccelSetLowPassFilter(char *cmdBuf, Imu_SensorId_e sID, int val, bool getSetting = false);

        int GyroRawOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        int GyroProcessedOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        int GyroAveOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        int GyroStdDevOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        int GyroSetFullScale(char *cmdBuf, Imu_SensorId_e sID, int val, bool getSetting = false);

        int GyroSetLowPassFilter(char *cmdBuf, Imu_SensorId_e sID, int val, bool getSetting = false);

        int BothAGRawOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        int BothAGProcessedOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        int BothAGAveOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        int BothAGStdDevOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        int BothAGTimeStatsEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting = false);

        int DataSkipNumber(char *cmdBuf, int val, bool getSetting = false);

        //Set the data output type to Base64 ('B' default) or Float 'F'
        //This is primarily used by a terminal program so that the data output
        //is in a human readable format
        int DataOutputType(char *cmdBuf, bool floattype, bool getSetting = false);

        int SetProcessType(char *cmdBuf, int val,  bool getSetting = false);

        int SetStatsProcNumberOfSamples(char *cmdBuf, int val,  bool getSetting = false);

        int SetParameterValue(char *cmdBuf, int paramSetNum, int idx, double pVal, bool getSetting);

        int CalibrationEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting);

    };

}
#endif //VIDERE_DEV_IMU_RXMESSAGEFORMATTER_H
