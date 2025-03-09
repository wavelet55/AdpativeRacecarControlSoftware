//
// Created by nvidia on 2/7/18.
//

#include "HostSystemInfo.h"
#include <iostream>
#include <memory>

namespace videre
{
    HostSystemInfo *HostSystemInfo::_hostSysInfo = nullptr;

    HostSystemInfo::HostSystemInfo()
    {
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
        Initialize();
    }

    HostSystemInfo * HostSystemInfo::getHostSystemInfoPtr()
    {
        if(HostSystemInfo::_hostSysInfo == nullptr)
        {
            HostSystemInfo::_hostSysInfo = new HostSystemInfo();
        }
        return HostSystemInfo::_hostSysInfo;
    }

    void HostSystemInfo::Initialize()
    {
        try
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

            //InitCpuPercentGather();
            //_sysinfo_static_msg->message_filled = true;
            //_sysinfo_static_msg->msg_str = StaticInformation();
            //_sysinfo_static_msg->PostMessage();

            LOGINFO("Boost Library Version: " << _libversions.Boost.to_string());
            LOGINFO("OpenCV Library Version: " << _libversions.OpenCV.to_string());
            LOGINFO("ZeroMQ Library Version: " << _libversions.ZeroMQ.to_string());
            LOGINFO("ProtoBuf Library Version: " << _libversions.Protobuf.to_string());
            if (_got_cpu_info)
                LOGINFO("CPU Info: " << _cpus[0].to_string());
            if(_got_gpu_info)
                LOGINFO("GPU Info: " << _gpus[0].to_string());
        }
        catch (std::exception &e)
        {
            LOGERROR("Error getting the Host Info: " << e.what());
        }
    }


    void HostSystemInfo::SetBoostVersion(LibraryVersions_t &v)
    {
        v.Boost.Major = BOOST_VERSION / 100000;
        v.Boost.Minor = BOOST_VERSION / 100 % 1000;
        v.Boost.Subminor = BOOST_VERSION % 100;
    }

    void HostSystemInfo::SetOpenCvVersion(LibraryVersions_t &v)
    {
        v.OpenCV.Major = CV_MAJOR_VERSION;
        v.OpenCV.Minor = CV_MINOR_VERSION;
        v.OpenCV.Subminor = CV_SUBMINOR_VERSION;
    }

    void HostSystemInfo::SetZeroMqVersion(LibraryVersions_t &v)
    {
        int major, minor, patch;
        zmq_version(&major, &minor, &patch);
        v.ZeroMQ.Major = major;
        v.ZeroMQ.Minor = minor;
        v.ZeroMQ.Subminor = patch;
    }

    void HostSystemInfo::SetProtobufVersion(LibraryVersions_t &v)
    {
        v.Protobuf.Major = GOOGLE_PROTOBUF_VERSION / 1000000;
        v.Protobuf.Minor = GOOGLE_PROTOBUF_VERSION / 1000 % 1000;
        v.Protobuf.Subminor = GOOGLE_PROTOBUF_VERSION % 100;
    }

    void HostSystemInfo::SetNetworkInfo(std::vector<NetworkInfo_t> &n)
    {
        struct ifaddrs *ifap, *ifa;
        struct sockaddr_in *sa;
        char *addr;

        try
        {

            getifaddrs(&ifap);
            for (ifa = ifap; ifa; ifa = ifa->ifa_next)
            {
                if (ifa->ifa_addr != NULL && ifa->ifa_addr->sa_family == AF_INET)
                {
                    sa = (struct sockaddr_in *) ifa->ifa_addr;
                    addr = inet_ntoa(sa->sin_addr);
                    n.push_back({ifa->ifa_name, addr});
                }
            }

            freeifaddrs(ifap);
        }
        catch (std::exception &e)
        {
            LOGERROR("Error getting the Host Network Info: " << e.what());
        }
    }

    bool HostSystemInfo::SetCPUInfo(std::vector<CpuInfo_t> &cpu)
    {
        int pindex = 0;
        std::string line;
        try
        {
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
            }
            else
            {
                _got_cpu_info = false;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("Error getting the Host Network Info: " << e.what());
        }
        return _got_cpu_info;
    }

    bool HostSystemInfo::SetMemoryInfo()
    {
        bool gotMemInfo = false;
        _total_memory = -1;
        std::string token;
        try
        {
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
                            gotMemInfo =  true;
                            break;
                        }
                        else
                        {
                            gotMemInfo = false;
                            break;
                        }
                    }
                    // ignore rest of the line
                    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
                file.close();
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("Error getting the Host Memory Info: " << e.what());
        }
        return gotMemInfo;
    }

    void HostSystemInfo::SetGpuInfo(std::vector<GpuInfo_t> &dgpu)
    {
        int gpu_count = 0;
        _got_gpu_info = false;
#ifdef CUDA
        try
        {
            int gpu_count = cv::cuda::getCudaEnabledDeviceCount();

            if (gpu_count > 0)
            {
                for (int n = 0; n < gpu_count; n++)
                {
                    cv::cuda::setDevice(0);
                    auto gpu_info = cv::cuda::DeviceInfo(n);
                    auto gpu = GpuInfo_t();
                    gpu.DeviceID = n;
                    gpu.Name = gpu_info.name();
                    gpu.MajorVersion = gpu_info.majorVersion();
                    gpu.MinorVersion = gpu_info.minorVersion();
                    gpu.MultiProcessorCount = gpu_info.multiProcessorCount();
                    gpu.TotalMemory = gpu_info.totalMemory();
                    gpu.IsCompatible = gpu_info.isCompatible();
                    gpu.IsUnifiedMemory = gpu_info.unifiedAddressing();
                    gpu.DynamicPrallelism = gpu_info.supports(cv::cuda::FeatureSet::DYNAMIC_PARALLELISM);
                    gpu.Compute_10 = gpu_info.supports(cv::cuda::FeatureSet::FEATURE_SET_COMPUTE_10);
                    gpu.Compute_11 = gpu_info.supports(cv::cuda::FeatureSet::FEATURE_SET_COMPUTE_11);
                    gpu.Compute_12 = gpu_info.supports(cv::cuda::FeatureSet::FEATURE_SET_COMPUTE_12);
                    gpu.Compute_13 = gpu_info.supports(cv::cuda::FeatureSet::FEATURE_SET_COMPUTE_13);
                    gpu.Compute_20 = gpu_info.supports(cv::cuda::FeatureSet::FEATURE_SET_COMPUTE_20);
                    gpu.Compute_21 = gpu_info.supports(cv::cuda::FeatureSet::FEATURE_SET_COMPUTE_21);
                    gpu.Compute_30 = gpu_info.supports(cv::cuda::FeatureSet::FEATURE_SET_COMPUTE_30);
                    gpu.Compute_35 = gpu_info.supports(cv::cuda::FeatureSet::FEATURE_SET_COMPUTE_35);
                    gpu.Compute_50 = gpu_info.supports(cv::cuda::FeatureSet::FEATURE_SET_COMPUTE_50);
                    gpu.GlobalAtomics = gpu_info.supports(cv::cuda::FeatureSet::GLOBAL_ATOMICS);
                    gpu.NativeDouble = gpu_info.supports(cv::cuda::FeatureSet::NATIVE_DOUBLE);
                    gpu.SharedAtomics = gpu_info.supports(cv::cuda::FeatureSet::SHARED_ATOMICS);
                    gpu.WarpShuffelFunction = gpu_info.supports(cv::cuda::FeatureSet::WARP_SHUFFLE_FUNCTIONS);

                    dgpu.push_back(gpu);
                }
                _got_gpu_info = true;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("Error getting the NVidia GPU Info: " << e.what());
        }
#endif
    }

    bool HostSystemInfo::SetCameraInfo()
    {
        bool gotCameraInfo = false;
        std::string token;
        try
        {
            std::ifstream file("/sys/class/video4linux/video0/name");
            if (file.is_open())
            {
                std::getline(file, token);
                _name_default_camera = token;
                file.close();
                gotCameraInfo = true;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("Error getting the Camera Info: " << e.what());
        }
        return gotCameraInfo;
    }

    //--------------------------Static JSON Output ---------------------------

    std::string HostSystemInfo::StaticInformation()
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

    std::string HostSystemInfo::VersionJson()
    {
        return _libversions.to_json();
    }

    std::string HostSystemInfo::NetworkJson()
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

    std::string HostSystemInfo::CPUJson()
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

    std::string HostSystemInfo::MemoryJson()
    {
        std::ostringstream os;
        os << "{\"TotalMemory\":" << _total_memory << "}";
        return os.str();
    }

    std::string HostSystemInfo::GPUJson()
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


}