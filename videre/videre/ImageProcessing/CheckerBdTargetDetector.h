/* ****************************************************************
 * Checker Board Target Detector
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: May 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#ifndef VIDERE_DEV_CHECKERBDTARGETDETECTOR_H
#define VIDERE_DEV_CHECKERBDTARGETDETECTOR_H

#include <RabitManager.h>
//#include "VehicleInertialStatesMessage.h"
//#include "CameraOrientationMessage.h"
//#include "image_plus_metadata_message.h"
#include "ImageProcTargetInfoResultsMessage.h"
#include "config_data.h"
#include "logger.h"
#include "global_defines.h"
#include <opencv2/core.hpp>
#include "TargetDetection/BlobTargetDetectorOpenCVSimple/StdBlobTargetDetectorOpenCVSimple.h"
#include "Utilities/ImagePixelLocationToRealWorldLocation.h"
#include "TargetParametersMessage.h"
#include "TargetDetector.h"
#include "CommonImageProcTypesDefs.h"
#include "TargetDetection/CheckerBoardTargetDetector/CheckerBoardTargetDetector.h"

#ifdef CUDA
#endif

using namespace GeoCoordinateSystemNS;
using namespace VidereImageprocessing;
using namespace ImageProcLibsNS;
using namespace MathLibsNS;

namespace TargetDetectorNS
{

    //This class is a shell for the primary Checker Board Target Detector
    //located in ImageProcessing Libraries.
    class CheckerBdTargetDetector : public TargetDetector
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;


        bool _cudaTargetDetectorEnabled = false;

        CheckerBoardTargetDetectorNS::CheckerBoardTargetDetector *_checkerBdTargetDetectorPtr = nullptr;


        double corner[4][2];

        //ImageMaskRect_t _mask;

        //ColorThresholds_t _colorThresholds;

    public:
        // Zoomed in or not?
        bool IsZoomedIn = false;

        // Using background modeling data?
        bool BackgroundModelingEnabled = false;

        // Removing sunlight reflected areas?
        bool SunlightReflectionEnabled = false;

        // Ignoring wheel occlusion?
        bool WheelRemovalEnabled = false;


        // Mark found targets in red on image if set
        bool MarkTargetEnabled = false;


    public:
        CheckerBdTargetDetector(Rabit::RabitManager* mgrPtr,
                           std::shared_ptr<ConfigData> config,
                            bool useGPGPUAcceleration);

        ~CheckerBdTargetDetector();

        //This flag will be true if useGPGPUAcceleration = true and
        //GPGPU is available, otherwise it will be false.
        bool IsGPGPUAccelerationEnabled()
        {
            return _cudaTargetDetectorEnabled;
        }

        virtual bool Initialize();

        virtual void Close();

        //Check and update target prameters if necessary.
        virtual void CheckTargetParameters();

        //Target Detection Method
        //Handles the Image Processing for detecting targets
        //and returns the results in the provide message.
        //Returns the number of targets found
        //A number less than zero indicates an error.
        virtual int DetectTargets(ImagePlusMetadataMessage* imagePlusMetaData,
                          ImageProcTargetInfoResultsMessage* targetResultsMsg,
                          std::shared_ptr<ImagePixelLocationToRealWorldLocation> pixeToRealWorldConvertor_sptr);


        int AddTargetsFoundToTargetList(std::vector<BlobTargetInfo_t> &tgtResults,
                                                                 int maxNoTgtsToAdd);

    };

}
#endif //VIDERE_DEV_CHECKERBDTARGETDETECTOR_H
