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

#ifndef VISION_SYSTEM_INFO
#define VISION_SYSTEM_INFO

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/version.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <zmq.hpp>
#include <google/protobuf/stubs/common.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <RabitManager.h>
#include "../../Messages/sysinfo_dynamic_message.h"
#include "../../Messages/sysinfo_static_message.h"
#include "../../Utils/global_defines.h"
#include "../../Utils/config_data.h"
#include "../../Utils/logger.h"

namespace videre
{

    class SystemInfoManager : public Rabit::RabitManager
    {

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
            bool DynamicPrallelism;
            bool Compute_10;
            bool Compute_11;
            bool Compute_12;
            bool Compute_13;
            bool Compute_20;
            bool Compute_21;
            bool Compute_30;
            bool Compute_35;
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
                   << "\"DynamicPrallelism\": " << DynamicPrallelism << ","
                   << "\"Compute_10\": " << Compute_10 << ","
                   << "\"Compute_11\": " << Compute_11 << ","
                   << "\"Compute_12\": " << Compute_12 << ","
                   << "\"Compute_13\": " << Compute_13 << ","
                   << "\"Compute_30\": " << Compute_30 << ","
                   << "\"Compute_35\": " << Compute_35 << ","
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
                DynamicPrallelism = false;
                Compute_10 = false;
                Compute_11 = false;
                Compute_12 = false;
                Compute_13 = false;
                Compute_20 = false;
                Compute_21 = false;
                Compute_30 = false;
                Compute_35 = false;
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

        /*************************************************************************
         *                            Members
         ************************************************************************/
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        LibraryVersions_t _libversions;
        std::vector<NetworkInfo_t> _network;
        std::vector<CpuInfo_t> _cpus;
        std::vector<GpuInfo_t> _gpus;
        int _total_memory = 0;
        std::string _name_default_camera;

        bool _got_cpu_info = false;
        bool _got_memory_info = false;

        bool _measuring_cpu_state;
        std::vector<double> _cpus_percent_used;
        std::vector<std::array<double, 4> > _cpu_stat1;
        std::vector<std::array<double, 4> > _cpu_stat2;
        unsigned int _free_memory;
        unsigned int _cache_memory;
        unsigned int _buffer_memory;
        unsigned int _measure_index;

        int _gpu_used_memory;

        std::shared_ptr<SysInfoDynamicMessage> _sysinfo_dynamic_msg;
        std::shared_ptr<SysInfoStaticMessage> _sysinfo_static_msg;

        /*************************************************************************
         *                            Methods
         ************************************************************************/
    public:

        SystemInfoManager(std::string name) : Rabit::RabitManager(name)
        {
            //Logger Setup
            log4cpp_ = log4cxx::Logger::getLogger("aobj");
            log4cpp_->setAdditivity(false);

            this->SetWakeupTimeDelayMSec(5000);  //Run ever 250 milliseconds

            _sysinfo_dynamic_msg = std::make_shared<SysInfoDynamicMessage>();
            this->AddPublishSubscribeMessage("SysInfoDynamicMessage", _sysinfo_dynamic_msg);

            _sysinfo_static_msg = std::make_shared<SysInfoStaticMessage>();
            this->AddPublishSubscribeMessage("SysInfoStaticMessage", _sysinfo_static_msg);
        }

        void Initialize()
        {
            _libversions = LibraryVersions_t();
            SetBoostVersion(_libversions);
            SetOpenCvVersion(_libversions);
            SetZeroMqVersion(_libversions);
            SetProtobufVersion(_libversions);
            SetNetworkInfo(_network);
            _got_cpu_info = SetCPUInfo(_cpus) ? true : false;
            _got_memory_info = SetMemoryInfo() ? true : false;
            SetGpuInfo(_gpus);
            SetCameraInfo();

            InitCpuPercentGather();
            _sysinfo_static_msg->message_filled = true;
            _sysinfo_static_msg->msg_str = StaticInformation();
            _sysinfo_static_msg->PostMessage();

            LOGINFO("Boost Library Version: " << _libversions.Boost.to_string());
            LOGINFO("OpenCV Library Version: " << _libversions.OpenCV.to_string());
            LOGINFO("ZeroMQ Library Version: " << _libversions.ZeroMQ.to_string());
            LOGINFO("ProtoBuf Library Version: " << _libversions.Protobuf.to_string());
            if( _got_cpu_info )
                LOGINFO("CPU Info: " << _cpus[0].to_string());
            //LOGINFO("GPU Info: " << _gpus[0].ToString());
        }

        void ExecuteUnitOfWork() final
        {
            GetCpuStatGather();
            GetUsedMemoryGather();
            _sysinfo_dynamic_msg->PostDynamicInfo(DynamicInformation());
            //std::cout << "Operational Info: " << _sysinfo_dynamic_msg->msg_str << std::endl;
        }

        //-------------------------------- Get Static Info -----------------------
        void SetBoostVersion(LibraryVersions_t &v)
        {
            v.Boost.Major = BOOST_VERSION / 100000;
            v.Boost.Minor = BOOST_VERSION / 100 % 1000;
            v.Boost.Subminor = BOOST_VERSION % 100;
        }

        void SetOpenCvVersion(LibraryVersions_t &v)
        {
            v.OpenCV.Major = CV_MAJOR_VERSION;
            v.OpenCV.Minor = CV_MINOR_VERSION;
            v.OpenCV.Subminor = CV_SUBMINOR_VERSION;
        }

        void SetZeroMqVersion(LibraryVersions_t &v)
        {
            int major, minor, patch;
            zmq_version(&major, &minor, &patch);
            v.ZeroMQ.Major = major;
            v.ZeroMQ.Minor = minor;
            v.ZeroMQ.Subminor = patch;
        }

        void SetProtobufVersion(LibraryVersions_t &v)
        {
            v.Protobuf.Major = GOOGLE_PROTOBUF_VERSION / 1000000;
            v.Protobuf.Minor = GOOGLE_PROTOBUF_VERSION / 1000 % 1000;
            v.Protobuf.Subminor = GOOGLE_PROTOBUF_VERSION % 100;
        }

        void SetNetworkInfo(std::vector<NetworkInfo_t> &n)
        {
            struct ifaddrs *ifap, *ifa;
            struct sockaddr_in *sa;
            char *addr;

            getifaddrs(&ifap);
            for (ifa = ifap; ifa; ifa = ifa->ifa_next)
            {
                if (ifa->ifa_addr->sa_family == AF_INET)
                {
                    sa = (struct sockaddr_in *) ifa->ifa_addr;
                    addr = inet_ntoa(sa->sin_addr);
                    n.push_back({ifa->ifa_name, addr});
                }
            }

            freeifaddrs(ifap);
        }

        bool SetCPUInfo(std::vector<CpuInfo_t> &cpu)
        {

            int pindex = 0;
            std::string line;
            std::ifstream myfile("/proc/cpuinfo");

            if (myfile.is_open())
            {
                std::getline(myfile, line);
                do
                {
                    auto scpu = CpuInfo_t();
                    while (!line.empty())
                    {
                        std::vector<std::string> results;
                        boost::split(results, line, boost::is_any_of(":"));

                        if (results.size() > 1)
                        {
                            if (line.compare(0, 10, "model name") == 0)
                                scpu.Model = results[1];

                            if (line.compare(0, 9, "vendor_id") == 0)
                                scpu.VendorID = results[1];

                            if (line.compare(0, 9, "processor") == 0)
                                scpu.Processor = stoi(results[1]);

                            if (line.compare(0, 9, "cpu cores") == 0)
                                scpu.CPUCores = stoi(results[1]);

                            if (line.compare(0, 7, "core id") == 0)
                                scpu.CoreID = stoi(results[1]);
                        }

                        std::getline(myfile, line);
                    }
                    cpu.push_back(scpu);

                    pindex++;

                } while (std::getline(myfile, line));

                myfile.close();
                _got_cpu_info = true;
                return true;
            } else
            {
                _got_cpu_info = false;
                return false;
            }
        }

        bool SetMemoryInfo()
        {

            _total_memory = -1;
            std::string token;
            std::ifstream file("/proc/meminfo");
            if (file.is_open())
            {
                while (file >> token)
                {
                    if (token == "MemTotal:")
                    {
                        unsigned long mem;
                        if (file >> mem)
                        {
                            _total_memory = mem;
                            return true;
                        } else
                        {
                            return false;
                        }
                    }
                    // ignore rest of the line
                    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
                file.close();
            } else
                return false;

        }

        void SetGpuInfo(std::vector<GpuInfo_t> &dgpu)
        {
            int gpu_count = 0;
            //int gpu_count = cv::gpu::getCudaEnabledDeviceCount();

            if (gpu_count > 0)
            {
                for (int n = 0; n < gpu_count; n++)
                {
                    //cv::gpu::setDevice(0);
                    //auto gpu_info = cv::gpu::DeviceInfo(n);
                    auto gpu = GpuInfo_t();
                    gpu.DeviceID = n;
                    //gpu.Name = gpu_info.name();
                    //gpu.MajorVersion = gpu_info.majorVersion();
                    //gpu.MinorVersion = gpu_info.minorVersion();
                    //gpu.MultiProcessorCount = gpu_info.multiProcessorCount();
                    //gpu.TotalMemory = gpu_info.totalMemory();
                    //gpu.IsCompatible = gpu_info.isCompatible();
                    //gpu.DynamicPrallelism = gpu_info.supports(cv::gpu::FeatureSet::DYNAMIC_PARALLELISM);
                    //gpu.Compute_10 = gpu_info.supports(cv::gpu::FeatureSet::FEATURE_SET_COMPUTE_10);
                    //gpu.Compute_11 = gpu_info.supports(cv::gpu::FeatureSet::FEATURE_SET_COMPUTE_11);
                    //gpu.Compute_12 = gpu_info.supports(cv::gpu::FeatureSet::FEATURE_SET_COMPUTE_12);
                    //gpu.Compute_13 = gpu_info.supports(cv::gpu::FeatureSet::FEATURE_SET_COMPUTE_13);
                    //gpu.Compute_20 = gpu_info.supports(cv::gpu::FeatureSet::FEATURE_SET_COMPUTE_20);
                    //gpu.Compute_21 = gpu_info.supports(cv::gpu::FeatureSet::FEATURE_SET_COMPUTE_21);
                    //gpu.Compute_30 = gpu_info.supports(cv::gpu::FeatureSet::FEATURE_SET_COMPUTE_30);
                    //gpu.Compute_35 = gpu_info.supports(cv::gpu::FeatureSet::FEATURE_SET_COMPUTE_35);
                    //gpu.GlobalAtomics = gpu_info.supports(cv::gpu::FeatureSet::GLOBAL_ATOMICS);
                    //gpu.NativeDouble = gpu_info.supports(cv::gpu::FeatureSet::NATIVE_DOUBLE);
                    //gpu.SharedAtomics = gpu_info.supports(cv::gpu::FeatureSet::SHARED_ATOMICS);
                    //gpu.WarpShuffelFunction = gpu_info.supports(cv::gpu::FeatureSet::WARP_SHUFFLE_FUNCTIONS);

                    dgpu.push_back(gpu);
                }
            }

        }

        bool SetCameraInfo()
        {

            std::string token;
            std::ifstream file("/sys/class/video4linux/video0/name");
            if (file.is_open())
            {
                std::getline(file, token);
                _name_default_camera = token;
                file.close();
            } else
                return false;
            return true;
        }

        //--------------------------Static JSON Output ---------------------------

        std::string StaticInformation()
        {
            std::ostringstream os;

            os << "{\"Versions\":" << VersionJson() << ","
               << "\"Network\":" << NetworkJson() << ","
               << "\"CPU\":" << CPUJson() << ","
               << "\"GPU\":" << GPUJson() << ","
               << "\"Memory\":" << MemoryJson() << ","
               << "\"Camera\":\"" << _name_default_camera << "\","
               << "\"VisionIPC\":" << "_vision_ipc_state" << "}";

            return os.str();
        }

        std::string VersionJson()
        {
            return _libversions.to_json();
        }

        std::string NetworkJson()
        {
            std::ostringstream os;
            os << "[";

            if (!_network.empty())
            {
                for (int n = 0; n < _network.size() - 1; n++)
                {
                    os << _network[n].to_json() << ",";
                }
                if (_network.size() > 0)
                    os << _network[_network.size() - 1].to_json();
            }
            os << "]";
            return os.str();
        }

        std::string CPUJson()
        {
            std::ostringstream os;
            os << "[";

            if (_got_cpu_info && !_cpus.empty())
            {
                for (int n = 0; n < _cpus.size() - 1; n++)
                {
                    os << _cpus[n].to_json() << ",";
                }
                if (_cpus.size() > 0)
                    os << _cpus[_cpus.size() - 1].to_json();
            }
            os << "]";
            return os.str();
        }

        std::string MemoryJson()
        {
            std::ostringstream os;
            os << "{\"TotalMemory\":" << _total_memory << "}";
            return os.str();
        }

        std::string GPUJson()
        {
            std::ostringstream os;
            os << "[";
            if (!_gpus.empty())
            {
                for (int n = 0; n < _gpus.size() - 1; n++)
                {
                    os << _gpus[n].to_json() << ",";
                }
                if (_gpus.size() > 0)
                    os << _gpus[_gpus.size() - 1].to_json();
            }
            os << "]";
            return os.str();
        }

        //--------------------------Dynamic Information---------------------------

        std::string DynamicInformation()
        {

            _measure_index++;
            std::ostringstream os;
            os << "{\"FG\":" << "{}" << ",\"BG\":"
               << "{\"id\":" << std::to_string(_measure_index)
               << ",\"CPU\":" + CpuStatToJson()
               << ",\"MEMORY\":" + UsedMemoryToJson() + "}";

            return os.str();
        }

        void InitCpuPercentGather()
        {
            _cpus_percent_used.clear();
            _cpu_stat1.clear();
            _cpu_stat2.clear();

            _cpu_stat1.push_back({0, 0, 0, 0});
            _cpu_stat2.push_back({0, 0, 0, 0});
            _cpus_percent_used.push_back(0);

            if (!_cpus.empty())
            {
                for (int n = 0; n < _cpus.size(); n++)
                {
                    _cpu_stat1.push_back({0, 0, 0, 0});
                    _cpu_stat2.push_back({0, 0, 0, 0});
                    _cpus_percent_used.push_back(0);
                }
            }

        }

        bool GetCpuStatGather()
        {
            const std::string cpuinfo = "/proc/stat";
            const int wait_millsec = 1000;
            std::ifstream myfile(cpuinfo);
            std::string ignore, a1, a2, a3, a4;
            std::string line;
            if (myfile.is_open() && !_cpus.empty())
            {

                for (int n = 0; n <= _cpus.size(); n++)
                {
                    std::getline(myfile, line);
                    std::istringstream(line) >> ignore >> a1 >> a2 >> a3 >> a4;
                    _cpu_stat1[n][0] = std::stod(a1);
                    _cpu_stat1[n][1] = std::stod(a2);
                    _cpu_stat1[n][2] = std::stod(a3);
                    _cpu_stat1[n][3] = std::stod(a4);
                }
                myfile.close();

                std::this_thread::sleep_for(std::chrono::milliseconds(wait_millsec));

                std::ifstream myfile2(cpuinfo);
                if (myfile2.is_open() && !_cpus.empty())
                {

                    for (int n = 0; n <= _cpus.size(); n++)
                    {
                        std::getline(myfile2, line);
                        std::istringstream(line) >> ignore >> a1 >> a2 >> a3 >> a4;
                        _cpu_stat2[n][0] = std::stod(a1);
                        _cpu_stat2[n][1] = std::stod(a2);
                        _cpu_stat2[n][2] = std::stod(a3);
                        _cpu_stat2[n][3] = std::stod(a4);
                    }
                    myfile2.close();

                    for (int n = 0; n <= _cpus.size(); n++)
                    {
                        auto sn2 = _cpu_stat2[n][0] + _cpu_stat2[n][1] + _cpu_stat2[n][2];
                        auto sn1 = _cpu_stat1[n][0] + _cpu_stat1[n][1] + _cpu_stat1[n][2];
                        auto sd2 = sn2 + _cpu_stat2[n][3];
                        auto sd1 = sn1 + _cpu_stat1[n][3];
                        _cpus_percent_used[n] = 100 * (sn2 - sn1) / (sd2 - sd1 + 0.0000000000001);
                    }

                    return true;
                }
            }
            return false;
        }

        std::string CpuStatToJson()
        {
            std::ostringstream os;
            if (!_cpus_percent_used.empty())
            {
                os << "{\"cm\":" << _cpus_percent_used[0] << ",";
                for (int n = 1; n < _cpus_percent_used.size() - 1; n++)
                {
                    os << "\"c" << n << "\":" << _cpus_percent_used[n] << ", ";
                }
                auto N = _cpus_percent_used.size() - 1;
                os << "\"c" << N << "\":" << _cpus_percent_used[N] << "}";
            } else
            {
                os << "{}";
            }
            return os.str();
        }

        bool GetUsedMemoryGather()
        {

            _free_memory = -1;
            _buffer_memory = -1;
            _cache_memory = -1;
            std::string token;
            std::ifstream file("/proc/meminfo");
            if (file.is_open())
            {
                while (file >> token)
                {
                    if (token == "MemFree:")
                    {
                        unsigned long mem;
                        if (file >> mem)
                        {
                            _free_memory = mem;
                        } else
                        {
                            return false;
                        }
                    }

                    if (token == "Buffers:")
                    {
                        unsigned long mem;
                        if (file >> mem)
                        {
                            _buffer_memory = mem;
                        } else
                        {
                            return false;
                        }
                    }

                    if (token == "Cached:")
                    {
                        unsigned long mem;
                        if (file >> mem)
                        {
                            _cache_memory = mem;
                            break;
                        } else
                        {
                            return false;
                        }
                    }
                    // ignore rest of the line
                    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
                file.close();
            } else
                return false;
        }

        std::string UsedMemoryToJson()
        {
            std::ostringstream os;
            os << "{\"FM\":" << _free_memory << ","
               << "\"BM\":" << _buffer_memory << ","
               << "\"CM\":" << _cache_memory << "}";

            return os.str();
        }
    };

}

#endif // VISION_SYSTEM_INFO
