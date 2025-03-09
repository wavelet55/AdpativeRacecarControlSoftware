/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: May. 2016
 *
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *******************************************************************/

#ifndef VIDERE_DEV_HOSTSYSTEMINFO_H
#define VIDERE_DEV_HOSTSYSTEMINFO_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/version.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <zmq.hpp>
#include <google/protobuf/stubs/common.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include "sysinfo_dynamic_message.h"
#include "sysinfo_static_message.h"
#include "global_defines.h"
#include "config_data.h"
#include "logger.h"


namespace videre
{
    //Singleton Global Class
    //Setup when Videre is started up.
    class HostSystemInfo
    {

    public:
        /*************************************************************************
         *                 Structures for holding system information
         ************************************************************************/
        struct Version_t
        {
            int Major;
            int Minor;
            int Subminor;

            std::string to_string()
            {
                std::ostringstream os;
                os << Major << "." << Minor << "." << Subminor;
                return os.str();
            }

            std::string to_json()
            {
                std::ostringstream os;
                os << "{\"Major\":" << Major << ","
                   << "\"Minor\":" << Minor << ","
                   << "\"Subminor\":" << Subminor << "}";
                return os.str();
            }

            void Clear()
            {
                Major = 0;
                Minor = 0;
                Subminor = 0;
            }
        };

        struct LibraryVersions_t
        {
            Version_t OpenCV;
            Version_t Boost;
            Version_t Protobuf;
            Version_t ZeroMQ;

            std::string to_json()
            {
                std::ostringstream os;
                os << "{\"OpenCV\":" << OpenCV.to_json() << ","
                   << "\"Boost\":" << Boost.to_json() << ","
                   << "\"Protobuf\":" << Protobuf.to_json() << ","
                   << "\"ZeroMQ\":" << ZeroMQ.to_json() << "}";
                return os.str();
            }

            void Clear()
            {
                OpenCV.Clear();
                Boost.Clear();
                Protobuf.Clear();
                ZeroMQ.Clear();
            }
        };

        struct CpuInfo_t
        {
            std::string VendorID;
            std::string Model;
            int Processor;
            int CPUCores;
            int CoreID;

            std::string to_string()
            {
                std::ostringstream os;
                os << VendorID << ", " << Model << ", " << Processor << ", "
                   << CPUCores << ", " << CoreID;
                return os.str();
            }

            std::string to_json()
            {
                std::ostringstream os;
                os << "{\"VendorID\":\"" << VendorID << "\","
                   << "\"Model\":\"" << Model << "\","
                   << "\"Processor\":" << Processor << ","
                   << "\"CPUCores\":" << CPUCores << ","
                   << "\"CoreID\":" << CoreID << "}";
                return os.str();
            }

            void Clear()
            {
                VendorID = "None";
                Model = "None";
                Processor = -1;
                CPUCores = -1;
                CPUCores = -1;
                CoreID = -1;
            }
        };

        struct GpuInfo_t
        {
            std::string Name;
            int MajorVersion;
            int MinorVersion;
            int MultiProcessorCount;
            int TotalMemory;
            int DeviceID;
            bool IsCompatible;
            bool IsUnifiedMemory;
            bool DynamicPrallelism;
            bool Compute_10;
            bool Compute_11;
            bool Compute_12;
            bool Compute_13;
            bool Compute_20;
            bool Compute_21;
            bool Compute_30;
            bool Compute_35;
            bool Compute_50;
            bool GlobalAtomics;
            bool NativeDouble;
            bool SharedAtomics;
            bool WarpShuffelFunction;

            std::string to_string()
            {
                std::ostringstream os;
                os << Name << ", Ver:" << MajorVersion << "."
                   << MinorVersion << ", MPs: "
                   << MultiProcessorCount << ", Mem: "
                   << TotalMemory / 1000000 << ", ID: "
                   << DeviceID << ", Comp: " << IsCompatible;
                return os.str();
            }

            std::string to_json()
            {
                std::ostringstream os;
                os << "{\"Name\": \"" << Name << "\","
                   << "\"MajorVersion\": " << MajorVersion << ","
                   << "\"MinorVersion\": " << MinorVersion << ","
                   << "\"MultiProcessorCount\": " << MultiProcessorCount << ","
                   << "\"TotalMemory\": " << TotalMemory << ","
                   << "\"DeviceID\": " << DeviceID << ","
                   << "\"IsCompatible\": " << IsCompatible << ","
                   << "\"IsUnifiedMemory\": " << IsUnifiedMemory << ","
                   << "\"DynamicPrallelism\": " << DynamicPrallelism << ","
                   //<< "\"Compute_10\": " << Compute_10 << ","
                   //<< "\"Compute_11\": " << Compute_11 << ","
                   //<< "\"Compute_12\": " << Compute_12 << ","
                   //<< "\"Compute_13\": " << Compute_13 << ","
                   << "\"Compute_30\": " << Compute_30 << ","
                   << "\"Compute_35\": " << Compute_35 << ","
                   << "\"Compute_50\": " << Compute_50 << ","
                   << "\"GlobalAtomics\": " << GlobalAtomics << ","
                   << "\"NativeDouble\": " << NativeDouble << ","
                   << "\"SharedAtomics\": " << SharedAtomics << ","
                   << "\"WarpShuffelFunction\": " << WarpShuffelFunction << "}";
                return os.str();
            }

            void Clear()
            {
                Name = "None";
                MajorVersion = 0;
                MinorVersion = 0;
                MultiProcessorCount = 0;
                TotalMemory = 0;
                DeviceID = -1;
                IsCompatible = false;
                IsUnifiedMemory = false;
                DynamicPrallelism = false;
                Compute_10 = false;
                Compute_11 = false;
                Compute_12 = false;
                Compute_13 = false;
                Compute_20 = false;
                Compute_21 = false;
                Compute_30 = false;
                Compute_35 = false;
                Compute_50 = false;
                GlobalAtomics = false;
                NativeDouble = false;
                SharedAtomics = false;
                WarpShuffelFunction = false;
            }
        };

        struct NetworkInfo_t
        {
            std::string Interface;
            std::string Address;

            std::string to_string()
            {
                std::ostringstream os;
                os << "Interface: " << Interface <<
                   ", Address: " << Address;
                return os.str();
            }

            std::string to_json()
            {
                std::ostringstream os;
                os << "{\"Interface\":\"" << Interface << "\","
                        "\"Address\":\"" << Address << "\"}";
                return os.str();
            }

            void Clear()
            {
                Interface = "None";
                Address = "None";
            }
        };



        static HostSystemInfo * getHostSystemInfoPtr();

        void Initialize();

        void SetBoostVersion(LibraryVersions_t &v);
        void SetOpenCvVersion(LibraryVersions_t &v);
        void SetZeroMqVersion(LibraryVersions_t &v);
        void SetProtobufVersion(LibraryVersions_t &v);
        void SetNetworkInfo(std::vector<NetworkInfo_t> &n);
        bool SetCPUInfo(std::vector<CpuInfo_t> &cpu);
        bool SetMemoryInfo();
        void SetGpuInfo(std::vector<GpuInfo_t> &dgpu);
        bool SetCameraInfo();
        std::string StaticInformation();
        std::string VersionJson();
        std::string NetworkJson();
        std::string CPUJson();
        std::string MemoryJson();
        std::string GPUJson();

    private:
        static HostSystemInfo *_hostSysInfo;

        HostSystemInfo();



        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        LibraryVersions_t _libversions;
        std::vector<NetworkInfo_t> _network;
        std::vector<CpuInfo_t> _cpus;
        std::vector<GpuInfo_t> _gpus;
        int _total_memory = 0;
        std::string _name_default_camera;
        bool _got_cpu_info = false;
        bool _got_gpu_info = false;
        bool _got_memory_info = false;

    };

}
#endif //VIDERE_DEV_HOSTSYSTEMINFO_H
