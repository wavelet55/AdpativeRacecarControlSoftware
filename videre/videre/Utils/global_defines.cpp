/* ****************************************************************
 * Global Definitions used across Videre
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: July 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#include "global_defines.h"
#include <iostream>
#include <sstream>
#include <string>
#include <boost/algorithm/string.hpp>


namespace videre
{

    std::string VisionSystemManagerNames[] =
            {
                    "UnknownManager",
                    "VidereSystemControlManager",
                    "HeadOrientationManager",
                    "VehicleStateManager",
                    "CommsManager",
                    "ImageCaptureManager",
                    "VisionProcessManager",
                    "StreamRecordManager",
                    "IMUCommManager",
                    "GPSManager",
                    "SipnPuffManager",
                    "VehicleActuatorInterfaceManager",
                    "RemoteControlManager"
                    "RobotArmManager",
                    "SystemInfoManager",
                    "DTX_IMU_InterfaceManager",
                    "EndOfManagerNames"
            };


    VisionSystemManagers_e GetVisionSystemManagerEnumFromName(const std::string& managerName)     {
        VisionSystemManagers_e mgr = VisionSystemManagers_e::VS_UnknownMgr;

        std::string mgrName = boost::algorithm::to_lower_copy(managerName);
        bool mgrNameFound = false;
        if (mgrName.compare("commsmanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_CommsManager;
        }
        else if (mgrName.compare("videresystemcontrolmanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_VidereSystemControlManager;
        }
        else if (mgrName.compare("headorientationmanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_HeadOrientationManager;
        }
        else if (mgrName.compare("vehiclestatemanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_VehicleStateManager;
        }
        else if (mgrName.compare("imagecapturemanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_ImageCaptureManager;
        }
        else if (mgrName.compare("streamrecordmanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_StreamRecordManager;
        }
        else if (mgrName.compare("visionprocessmanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_VisionProcessManager;
        }
        else if (mgrName.compare("systeminfomanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_SystemInfoManager;
        }
        else if (mgrName.compare("imucommmanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_IMUCommManager;
        }
        else if (mgrName.compare("dtx_imu_interfacemanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_DTX_IMU_InterfaceManager;
        }
        else if (mgrName.compare("gpsmanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_GPSManager;
        }
        else if (mgrName.compare("robotarmmanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_RobotArmManager;
        }
        else if (mgrName.compare("sipnpuffmanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_SipnPuffManager;
        }
        else if (mgrName.compare("vehicleactuatorinterfacemanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_VehicleActuatorInterfaceManager;
        }
        else if (mgrName.compare("remotecontrolmanager") == 0)
        {
            mgr = VisionSystemManagers_e::VS_RemoteControlManager;
        }

        return mgr;
    }

    bool AddDateTimeToDataLogDirectory = true;
    std::string DataLogDirectory = "DataLogs";

    std::string MakeManagerPlusStringName(VisionSystemManagers_e mgr, const std::string& strName)
    {
        std::ostringstream os;
        int idx = (int)mgr;
        idx = idx < 0 ? 0 : idx > VisionSystemNumberOfManagers - 1 ? VisionSystemNumberOfManagers - 1 : idx;
        os << VisionSystemManagerNames[(int) mgr] << ":" << strName;
        return os.str();
    }

}