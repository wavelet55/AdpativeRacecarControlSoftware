/* ****************************************************************
 * Blob Target Detector
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Nov. 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
   *******************************************************************/

#include "TargetDetectorProcessControl.h"
#include "BlobTargetDetectionProcess/BlobTargetDetector.h"
#include "CheckerBdTargetDetector.h"

namespace TargetDetectorNS
{

    TargetDetectorProcessControl::TargetDetectorProcessControl(Rabit::RabitManager* mgrPtr,
                                            std::shared_ptr<ConfigData> config)
    {
        _mgrPtr = mgrPtr;
        _config_sptr = config;

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
        LOGINFO("TargetDetectorProcessControl Created.");

        _activeTargetDetectorPtr = nullptr;

        for(int i = 0; i < NoTgtDetectAlgos; i++)
        {
            TgtDectAlgArray[i].stdAlgPtr = nullptr;
            TgtDectAlgArray[i].gpgpuAlgPtr = nullptr;
        }

        //Instantiate and Load All of the available Target Detector Processors.

        //Blob Target Detector
        //OpenCV Blob Detector Lib.
        TgtDectAlgArray[0].stdAlgPtr = new BlobTargetDetector(_mgrPtr, _config_sptr, true, false);
        ////Blob detector using Inspecta S.L. Lib.
        TgtDectAlgArray[1].stdAlgPtr = new BlobTargetDetector(_mgrPtr, _config_sptr, false, false);

        TgtDectAlgArray[2].stdAlgPtr = new CheckerBdTargetDetector(_mgrPtr, _config_sptr, false);
        TgtDectAlgArray[2].stdAlgAvailible = true;

#ifdef CUDA
        TgtDectAlgArray[0].gpgpuAlgPtr = new BlobTargetDetector(_mgrPtr, _config_sptr, true, true);
        TgtDectAlgArray[1].gpgpuAlgPtr = new BlobTargetDetector(_mgrPtr, _config_sptr, false, true);
#endif  //CUDA

        //This is the Standard / Default Target Detector.
        _activeTargetDetectorPtr = TgtDectAlgArray[0].stdAlgPtr;
        _activeTargetDetectorPtr->Initialize();
        _activeTargetProcessingMode = TargetProcessingMode_e::TgtProcMode_Blob;

    }


    TargetDetectorProcessControl::~TargetDetectorProcessControl()
    {
        for(int i = 0; i < NoTgtDetectAlgos; i++)
        {
            if(TgtDectAlgArray[i].stdAlgPtr != nullptr)
            {
                TgtDectAlgArray[i].stdAlgPtr->Close();
                delete(TgtDectAlgArray[i].stdAlgPtr);
            }
            if(TgtDectAlgArray[i].gpgpuAlgPtr != nullptr)
            {
                TgtDectAlgArray[i].gpgpuAlgPtr->Close();
                delete(TgtDectAlgArray[i].gpgpuAlgPtr);
            }
        }
    }

    //Set the Active Target Processing Mode / Algorithm.
    //By default the GPGPU Accelerated version of the Target
    //Detector will be choosen if available unless the
    //forceNonGPGPUAlgorithm = true;  The forceNonGPGPUAlgorithm flag
    //is used to support comparing performace of accelerated and standard
    //algorithms.  The Intialized() method of the target detection
    //object is automatically called.
    //Returns true if there was an error setting up the algorithm,
    //false if it setup ok.
    bool TargetDetectorProcessControl::SetTargetProcessingMode(TargetProcessingMode_e tgtProcMode,
                                                                bool useGPGPUAlgorithm)
    {
        bool error = false;
        if( tgtProcMode != _activeTargetProcessingMode
                || _GPGPUAcceleratedAlgorithm != useGPGPUAlgorithm )
        {
            //First release any resources used by the current target detector.
            if(_activeTargetDetectorPtr != nullptr)
            {
                _activeTargetDetectorPtr->ReleaseResources();
            }
            switch (tgtProcMode)
            {
                case TargetProcessingMode_e::TgtProcMode_None:
                    _activeTargetDetectorPtr = TgtDectAlgArray[0].stdAlgPtr;
                    _GPGPUAcceleratedAlgorithm = false;
                    _activeTargetProcessingMode = TargetProcessingMode_e::TgtProcMode_None;
                    cout << "TgtProcMode:None chosen." << endl;
                    LOGINFO("TgtProcMode:None chosen.");
                    break;

                case TargetProcessingMode_e::TgtProcMode_Std:
                    if (!useGPGPUAlgorithm || TgtDectAlgArray[0].gpgpuAlgPtr == nullptr)
                    {
                        _activeTargetDetectorPtr = TgtDectAlgArray[0].stdAlgPtr;
                        _GPGPUAcceleratedAlgorithm = false;
                        cout << "TgtProcMode:Std (Standard OpenCv:Blob) chosen." << endl;
                        LOGINFO("TgtProcMode:Std (Standard OpenCv:Blob) chosen.");
                    }
                    else
                    {
                        _activeTargetDetectorPtr = TgtDectAlgArray[0].gpgpuAlgPtr;
                        _GPGPUAcceleratedAlgorithm = true;
                        cout << "TgtProcMode:Std (Cuda GPU OpenCV:Blob) chosen." << endl;
                        LOGINFO("TgtProcMode:Std (Cuda GPU OpenCV:Blob) chosen.");
                    }
                    _activeTargetProcessingMode = TargetProcessingMode_e::TgtProcMode_Std;
                    break;

                case TargetProcessingMode_e::TgtProcMode_Blob:
                    if (!useGPGPUAlgorithm || TgtDectAlgArray[1].gpgpuAlgPtr == nullptr)
                    {
                        _activeTargetDetectorPtr = TgtDectAlgArray[1].stdAlgPtr;
                        _GPGPUAcceleratedAlgorithm = false;
                        cout << "TgtProcMode:Blob (Standard) chosen." << endl;
                        LOGINFO("TgtProcMode:Blob (Standard) chosen.");
                    }
                    else
                    {
                        _activeTargetDetectorPtr = TgtDectAlgArray[1].gpgpuAlgPtr;
                        _GPGPUAcceleratedAlgorithm = true;
                        cout << "TgtProcMode:Blob (Cuda GPU) chosen." << endl;
                        LOGINFO("TgtProcMode:Blob (Cuda GPU) chosen.");
                    }
                    _activeTargetProcessingMode = TargetProcessingMode_e::TgtProcMode_Blob;
                    break;

                case TargetProcessingMode_e::TgtProcMode_CheckerBoard:
                    _activeTargetDetectorPtr = TgtDectAlgArray[2].stdAlgPtr;
                    _GPGPUAcceleratedAlgorithm = false;
                    cout << "TgtProcMode:Checker Board chosen." << endl;
                    LOGINFO("TgtProcMode:Checker Board chosen.");
                    break;

                default:
                    _activeTargetDetectorPtr = TgtDectAlgArray[0].stdAlgPtr;
                    _GPGPUAcceleratedAlgorithm = false;
                    _activeTargetProcessingMode = TargetProcessingMode_e::TgtProcMode_None;
                    cout << "TgtProcMode:None chosen." << endl;
                    LOGWARN("SetTargetProcessingMode:Default: TgtProcMode:None chosen.");
            }
            _activeTargetDetectorPtr->Initialize();
            _activeTargetDetectorPtr->CheckUpdateTargetParameters(true);
        }
        return error;
    }



}