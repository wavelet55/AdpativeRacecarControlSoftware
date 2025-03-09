/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/

#include "IMU_TxMessageFormatter.h"
#include "Base64Conversions.h"
#include <stdlib.h>
#include <string.h>

using namespace Rabit;

namespace IMU_SensorNS
{

    int IMU_TxMessageFormatter::buildCmdWithSensorID(char *cmdBuf, const char *cmd, Imu_SensorId_e sID)
    {
        int cIdx = cpyStringToBuf(cmdBuf, 0, IMU_MAXMESSAGESIZE, cmd, false);
        cmdBuf[cIdx++] = (int)sID + '0';
        cmdBuf[cIdx++] = '=';
        cmdBuf[cIdx] = 0;
        return cIdx;
    }

    int IMU_TxMessageFormatter::buildCmd(char *cmdBuf, const char *cmd)
    {
        int cIdx = cpyStringToBuf(cmdBuf, 0, IMU_MAXMESSAGESIZE, cmd, false);
        cmdBuf[cIdx++] = '=';
        cmdBuf[cIdx] = 0;
        return cIdx;
    }

    int IMU_TxMessageFormatter::addOnOffStr(char *cmdBuf, int offset, bool onOff, bool termCmd)
    {
        int cIdx = 0;
        if(onOff)
            cIdx = cpyStringToBuf(cmdBuf, offset, IMU_MAXMESSAGESIZE, "ON", false);
        else
            cIdx = cpyStringToBuf(cmdBuf, offset, IMU_MAXMESSAGESIZE, "OFF", false);
        if(termCmd)
            cIdx = addCommandTermination(cmdBuf + cIdx);
        return cIdx;
    }


    int IMU_TxMessageFormatter::AccelGyroEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "AGE", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    int IMU_TxMessageFormatter::AccelGyroDataRateDivider(char *cmdBuf, Imu_SensorId_e sID, int val, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "AGR", sID);
        if(!getSetting)
        {
            cIdx += sprintf(cmdBuf + cIdx, "%d", val);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }



    int IMU_TxMessageFormatter::AccelRawOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "AOR", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }


    int IMU_TxMessageFormatter::AccelProcessedOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "AOP", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }


    int IMU_TxMessageFormatter::AccelAveOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "AOA", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    int IMU_TxMessageFormatter::AccelStdDevOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "AOS", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    int IMU_TxMessageFormatter::AccelSetFullScale(char *cmdBuf, Imu_SensorId_e sID, int val, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "AFS", sID);
        if(!getSetting)
        {
            cIdx += sprintf(cmdBuf + cIdx, "%d", val);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }


    int IMU_TxMessageFormatter::AccelSetLowPassFilter(char *cmdBuf, Imu_SensorId_e sID, int val, bool getSetting )
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "ALP", sID);
        if(!getSetting)
        {
            cIdx += sprintf(cmdBuf + cIdx, "%d", val);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    int IMU_TxMessageFormatter::GyroRawOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "GOR", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }


    int IMU_TxMessageFormatter::GyroProcessedOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "GOP", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }


    int IMU_TxMessageFormatter::GyroAveOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "GOA", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    int IMU_TxMessageFormatter::GyroStdDevOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "GOS", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    int IMU_TxMessageFormatter::GyroSetFullScale(char *cmdBuf, Imu_SensorId_e sID, int val, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "GFS", sID);
        if(!getSetting)
        {
            cIdx += sprintf(cmdBuf + cIdx, "%d", val);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }


    int IMU_TxMessageFormatter::GyroSetLowPassFilter(char *cmdBuf, Imu_SensorId_e sID, int val, bool getSetting )
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "GLP", sID);
        if(!getSetting)
        {
            cIdx += sprintf(cmdBuf + cIdx, "%d", val);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }


    int IMU_TxMessageFormatter::BothAGRawOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "BOR", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }


    int IMU_TxMessageFormatter::BothAGProcessedOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "BOP", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }


    int IMU_TxMessageFormatter::BothAGAveOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "BOA", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    int IMU_TxMessageFormatter::BothAGStdDevOutputEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "BOS", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    int IMU_TxMessageFormatter::BothAGTimeStatsEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "BTS", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    int IMU_TxMessageFormatter::DataSkipNumber(char *cmdBuf, int val, bool getSetting)
    {
        int cIdx = buildCmd(cmdBuf, "DSN");
        if(!getSetting)
        {
            cIdx += sprintf(cmdBuf + cIdx, "%d", val);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    //Set the data output type to Base64 ('B' default) or Float 'F'
    //This is primarily used by a terminal program so that the data output
    //is in a human readable format
    int IMU_TxMessageFormatter::DataOutputType(char *cmdBuf, bool floattype, bool getSetting)
    {
        int cIdx = buildCmd(cmdBuf, "DOT");
        if(!getSetting)
        {
            cmdBuf[cIdx++] = floattype ? 'F' : 'B';
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    int IMU_TxMessageFormatter::SetProcessType(char *cmdBuf, int val,  bool getSetting)
    {
        int cIdx = buildCmd(cmdBuf, "PRC");
        if(!getSetting)
        {
            cIdx += sprintf(cmdBuf + cIdx, "%d", val);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    int IMU_TxMessageFormatter::SetStatsProcNumberOfSamples(char *cmdBuf, int val,  bool getSetting)
    {
        int cIdx = buildCmd(cmdBuf, "PSN");
        if(!getSetting)
        {
            cIdx += sprintf(cmdBuf + cIdx, "%d", val);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    int IMU_TxMessageFormatter::SetParameterValue(char *cmdBuf, int paramSetNum, int idx, double pVal, bool getSetting)
    {
        int cIdx = 0;
        if(!getSetting)
        {
            cIdx += sprintf(cmdBuf + cIdx, "PSF%d=%d,%f", paramSetNum, idx, pVal);
        }
        else
        {
            cIdx += sprintf(cmdBuf + cIdx, "PSF%d=%d,?", paramSetNum, idx);
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

    int IMU_TxMessageFormatter::CalibrationEnable(char *cmdBuf, Imu_SensorId_e sID, bool enable, bool getSetting)
    {
        int cIdx = buildCmdWithSensorID(cmdBuf, "CEN", sID);
        if(!getSetting)
        {
            cIdx = addOnOffStr(cmdBuf, cIdx, enable, false);
        }
        else
        {
            cmdBuf[cIdx++] = '?';
        }
        return addCommandTermination(cmdBuf + cIdx);
    }

}