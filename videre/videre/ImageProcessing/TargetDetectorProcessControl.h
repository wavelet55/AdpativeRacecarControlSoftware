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

#ifndef VIDERE_DEV_TARGETDETECTORPROCESSCONTROL_H
#define VIDERE_DEV_TARGETDETECTORPROCESSCONTROL_H

#include <string>
#include <RabitManager.h>
#include "TargetDetector.h"
#include "logger.h"
#include "config_data.h"

namespace TargetDetectorNS
{

    //This class keeps tract of the availible Target Detector
    //Processors and aids in the selection of the current
    //Target Detector... it is assumed that only one target
    //detector process/algorithm will be used at any given time.
    //Target Detectors may have GPGPU Accelerated versions and
    //standerd, non-accelerated versions if the hardware platform
    //does not support a GPGPU.
    class TargetDetectorProcessControl
    {
    private:
        const static int NoTgtDetectAlgos = 3;

        std::shared_ptr<ConfigData> _config_sptr;

        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        //The reference to the manager is used primarily for
        //setup purposes.
        Rabit::RabitManager* _mgrPtr;

        TargetDetector *_activeTargetDetectorPtr;

        TargetProcessingMode_e _activeTargetProcessingMode;

        bool _GPGPUAcceleratedAlgorithm = false;

        struct TgtDectAlgStruct
        {
            TargetDetector *stdAlgPtr;
            TargetDetector *gpgpuAlgPtr;
            bool stdAlgAvailible;
            bool gpgpuAlgAvailible;
        };


        TgtDectAlgStruct TgtDectAlgArray[NoTgtDetectAlgos];

    public:
        TargetDetectorProcessControl(Rabit::RabitManager* mgrPtr,
                                     std::shared_ptr<ConfigData> config);

        ~TargetDetectorProcessControl();

        TargetDetector* GetActiveTargetDetector()
        {
            return _activeTargetDetectorPtr;
        }

        TargetProcessingMode_e GetActiveTargetProcessingMode()
        {
            return _activeTargetProcessingMode;
        }

        bool IsGPUAcceleratedAlgorithm()
        {
            return _GPGPUAcceleratedAlgorithm;
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
        bool SetTargetProcessingMode(TargetProcessingMode_e tgtProcMode,
                                        bool useGPGPUAlgorithm = true);

    };

}
#endif //VIDERE_DEV_TARGETDETECTORPROCESSCONTROL_H
