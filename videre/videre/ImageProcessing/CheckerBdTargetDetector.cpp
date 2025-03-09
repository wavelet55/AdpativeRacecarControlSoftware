/* ****************************************************************
 * Blob Target Detector
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

#include "CheckerBdTargetDetector.h"
#include <math.h>
#include <GeoCoordinateSystem.h>


using namespace std;
using namespace cv;
using namespace VidereImageprocessing;
using namespace GeoCoordinateSystemNS;
using namespace CheckerBoardTargetDetectorNS;


namespace TargetDetectorNS
{

    CheckerBdTargetDetector::CheckerBdTargetDetector(Rabit::RabitManager* mgrPtr,
                                           std::shared_ptr<ConfigData> config,
                                           bool useGPGPUAcceleration)
        :TargetDetector(mgrPtr, config)
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        _checkerBdTargetDetectorPtr = new CheckerBoardTargetDetector();
        _cudaTargetDetectorEnabled = false;
        LOGINFO("CheckerBdTargetDetector: Created.");
        cout << "CheckerBdTargetDetector: Created." << endl;
    }

    CheckerBdTargetDetector::~CheckerBdTargetDetector()
    {
        if(_checkerBdTargetDetectorPtr != nullptr)
        {
            _checkerBdTargetDetectorPtr->Close();
            delete(_checkerBdTargetDetectorPtr);
        }
    }


    bool CheckerBdTargetDetector::Initialize()
    {
        bool error = false;
        if(_checkerBdTargetDetectorPtr != nullptr)
        {
            error = _checkerBdTargetDetectorPtr->Initialize();
            _targetType1ParamsMsg->FetchMessage();
            _checkerBdTargetDetectorPtr->Set_NumberOfObjects_X_Axis((int)_targetType1ParamsMsg->TargetSizeOrWidth);
            _checkerBdTargetDetectorPtr->Set_NumberOfObjects_Y_Axis((int)_targetType1ParamsMsg->TargetPerimeterOrLenght);
            _checkerBdTargetDetectorPtr->Set_NumberOfObjects_X_Axis(6);
            _checkerBdTargetDetectorPtr->Set_NumberOfObjects_Y_Axis(9);
            _checkerBdTargetDetectorPtr->MarkTargetEnabled = true;
        }
        return error;
    }

    void CheckerBdTargetDetector::Close()
    {
        if(_checkerBdTargetDetectorPtr != nullptr)
        {
            _checkerBdTargetDetectorPtr->Close();
        }
    }

    //Check and update target prameters if necessary.
    void CheckerBdTargetDetector::CheckTargetParameters()
    {
        if( _targetType1ParamsMsg->FetchMessage() )
        {
            //ToDo: Set base upon target parameters.
            //_checkerBdTargetDetectorPtr->Set_NumberOfObjects_X_Axis((int)_targetType1ParamsMsg->TargetSizeOrWidth);
            //_checkerBdTargetDetectorPtr->Set_NumberOfObjects_Y_Axis((int)_targetType1ParamsMsg->TargetPerimeterOrLenght);
            _checkerBdTargetDetectorPtr->Set_NumberOfObjects_X_Axis(6);
            _checkerBdTargetDetectorPtr->Set_NumberOfObjects_Y_Axis(9);
            _checkerBdTargetDetectorPtr->MarkTargetEnabled = true;
        }
    }


    //Target Detection Method
    //Handles the Image Processing for detecting targets
    //and returns the results in the provide message.
    //Returns the number of targets found
    //A number less than zero indicates an error.
    int CheckerBdTargetDetector::DetectTargets(ImagePlusMetadataMessage* imagePlusMetaData,
                                          ImageProcTargetInfoResultsMessage* targetResultsMsg,
                                          std::shared_ptr<ImagePixelLocationToRealWorldLocation> pixeToRealWorldConvertor_sptr)
    {
        int numberOfTargetsFound = 0;
        std::vector<BlobTargetInfo_t> tgtResults;

        _targetResultsMsg = targetResultsMsg;
        _pixeToRealWorldConvertor = pixeToRealWorldConvertor_sptr;

        //First check to ensure the UAV is at a reasonable altitude and
        //bank angle... otherwise the image processing will be meaningless.
        if(imagePlusMetaData->VehicleInertialStates.HeightAGL < ALT_UAV_LOW
                || imagePlusMetaData->VehicleInertialStates.HeightAGL > ALT_UAV_HIGH)
        {
            LOGINFO("CheckerBdTargetDetector: UAV Altitude is out of range.  Alt(AGL)=?" );
            //targetInfo->ImageProcStatus = AltitudeOutOfRange;
            return -1;
        }

        if(imagePlusMetaData->VehicleInertialStates.RollPitchYaw.PitchRadians() < PITCH_LOW
           || imagePlusMetaData->VehicleInertialStates.RollPitchYaw.PitchRadians() > PITCH_HIGH)
        {
            LOGINFO("CheckerBdTargetDetector: UAV Pitch is out of range." );
            //targetInfo->ImageProcStatus = AltitudeOutOfRange;
            return -1;
        }

        if(imagePlusMetaData->VehicleInertialStates.RollPitchYaw.RollRadians() < ROLL_LOW
           || imagePlusMetaData->VehicleInertialStates.RollPitchYaw.RollRadians() > ROLL_HIGH)
        {
            LOGINFO("CheckerBdTargetDetector: UAV Roll is out of range." );
            //targetInfo->ImageProcStatus = AltitudeOutOfRange;
            return -1;
        }

        if(imagePlusMetaData->CameraOrientation.CameraAzimuthElevationAngles.ElevationAngleRad() > TASE_HIGH )
        {
            LOGINFO("CheckerBdTargetDetector: Camera Elevation Angle is out of range." );
            //targetInfo->ImageProcStatus = AltitudeOutOfRange;
            return -1;
        }

        //Get a reference to the Sensor Image passed into the method.
        //imgInpRGB = &imagePlusMetaData->ImageFrame;
        if(_checkerBdTargetDetectorPtr != nullptr)
        {
            numberOfTargetsFound = _checkerBdTargetDetectorPtr->DetectTargets(&imagePlusMetaData->ImageFrame, &tgtResults);
        }

        if(numberOfTargetsFound > 0)
        {
            AddTargetsFoundToTargetList(tgtResults, 4);
        }

        return numberOfTargetsFound;
    }




    int CheckerBdTargetDetector::AddTargetsFoundToTargetList(std::vector<BlobTargetInfo_t> &tgtResults,
                                        int maxNoTgtsToAdd)
    {
        int NoTgtsAdded = 0;
        XYZCoord_t xyzCoord;
        LatLonAltCoord_t latLonAltCoord;
        AzimuthElevation_t azimuthElevation;
        GeoCoordinateSystem *gcsPtr;
        maxNoTgtsToAdd = maxNoTgtsToAdd > MAX_NUMBER_TARGETS_PER_IMAGE ? MAX_NUMBER_TARGETS_PER_IMAGE : maxNoTgtsToAdd;

        int Nx = _checkerBdTargetDetectorPtr->Get_NumberOfObjects_X_Axis();
        int Ny = _checkerBdTargetDetectorPtr->Get_NumberOfObjects_Y_Axis();
        int N = Nx * Ny;
        int Ntgts = tgtResults.size();

        int idxArray[4];
        if( Ntgts >= N && Ntgts >= 4)
        {
            idxArray[0] = 0;
            idxArray[1] = Nx - 1;
            idxArray[2] = Ntgts - Nx;
            idxArray[3] = Ntgts - 1;
        }
        else
        {
            idxArray[0] = 0;
            idxArray[1] = 1;
            idxArray[2] = 2;
            idxArray[3] = 3;
        }

        N = Ntgts > 4 ? 4 : Ntgts;

        if( _targetResultsMsg != nullptr )
        {
            for(int i = 0; i < N; i++)
            {
                int idx = idxArray[i];
                double blob_XPix = tgtResults[idx].TgtCenterPixel_x;
                double blob_YPix = tgtResults[idx].TgtCenterPixel_y;
                double orientationRad = tgtResults[idx].TgtOrientationAngleDeg;

                if (_pixeToRealWorldConvertor != nullptr)
                {
                    _pixeToRealWorldConvertor->CalculateRealWorldLocation(blob_XPix, blob_YPix,
                                                                          &xyzCoord,
                                                                          &azimuthElevation);
                    gcsPtr = GeoCoordinateSystem::GetGeoCoordinateSystemReference();
                    latLonAltCoord = gcsPtr->XYZToLatLonAlt(xyzCoord);
                }

                if( _targetResultsMsg->AddTarget(1,
                                             blob_XPix, blob_YPix,
                                             orientationRad,
                                             latLonAltCoord,
                                             azimuthElevation) )
                {
                    LOGERROR("AddTargetsFoundToTargetList: Error adding target to Target Info Message")
                }
            }
        }

        return NoTgtsAdded;
    }

}