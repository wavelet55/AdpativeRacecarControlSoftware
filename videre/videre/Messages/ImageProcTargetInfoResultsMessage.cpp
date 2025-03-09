/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: March. 2016
 *
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *******************************************************************/

#include "ImageProcTargetInfoResultsMessage.h"

namespace videre
{

    void ImageProcTargetInfoResultsMessage::Clear()
    {
        TargetInfoResultsPBMsg.clear_targetlocations();
    }

    std::unique_ptr<Rabit::RabitMessage> ImageProcTargetInfoResultsMessage::Clone() const
    {
        auto clone = std::unique_ptr<ImageProcTargetInfoResultsMessage>(new ImageProcTargetInfoResultsMessage(*this));
        return std::move(clone);
    }

    bool ImageProcTargetInfoResultsMessage::CopyMessage(Rabit::RabitMessage *msg)
    {
        Rabit::RabitMessage::CopyMessage(msg); // call baseclass
        if (msg->GetTypeIndex() == std::type_index(typeid(ImageProcTargetInfoResultsMessage)))
        {
            //Ensure the Copy process does not loose the mesages'
            //publish subscribe reference.
            std::shared_ptr<Rabit::PublishSubscribeMessage> psmr = this->GetGlobalPublishSubscribeMessageRef();

            ImageProcTargetInfoResultsMessage *coMsg = static_cast<ImageProcTargetInfoResultsMessage *>(msg);
            *this = *coMsg;

            this->SetGlobalPublishSubscribeMessageRef(psmr);
            return true;
        }
        return false;
    }

    void ImageProcTargetInfoResultsMessage::SetImageMetaDataInfo(const ImagePlusMetadataMessage *imgMetaData)
    {
        //TargetInfoResultsPBMsg.clear_vehicleinertialstates();
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_coordinateslatlonorxy(true);
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_latituderadory(
                imgMetaData->VehicleInertialStates.LatLonAlt.LatitudeRadians());
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_longituderadorx(
                imgMetaData->VehicleInertialStates.LatLonAlt.LongitudeRadians());
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_altitudemsl(
                imgMetaData->VehicleInertialStates.LatLonAlt.Altitude());
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_heightagl(
                imgMetaData->VehicleInertialStates.HeightAGL);
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_veleastmps(
                imgMetaData->VehicleInertialStates.XYZVelocities.x);
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_velnorthmps(
                imgMetaData->VehicleInertialStates.XYZVelocities.y);
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_veldownmps(
                imgMetaData->VehicleInertialStates.XYZVelocities.z);
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_rollrad(
                imgMetaData->VehicleInertialStates.RollPitchYaw.RollRadians());
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_pitchrad(
                imgMetaData->VehicleInertialStates.RollPitchYaw.PitchRadians());
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_yawrad(
                imgMetaData->VehicleInertialStates.RollPitchYaw.YawRadians());
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_rollrateradps(
                imgMetaData->VehicleInertialStates.RollPitchYawRates.RollRadians());
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_pitchrateradps(
                imgMetaData->VehicleInertialStates.RollPitchYawRates.PitchRadians());
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_yawrateradps(
                imgMetaData->VehicleInertialStates.RollPitchYawRates.YawRadians());
        TargetInfoResultsPBMsg.mutable_vehicleinertialstates()->set_gpstimestampsec(
                imgMetaData->VehicleInertialStates.GpsTimeStampSec);

        //Set Image Location Info
        //TargetInfoResultsPBMsg.clear_imagelocation();
        TargetInfoResultsPBMsg.mutable_imagelocation()->set_imagenumber(imgMetaData->ImageNumber);
        TargetInfoResultsPBMsg.mutable_imagelocation()->set_targetcornerslatlonorxy(false);
        if (TargetInfoResultsPBMsg.imagelocation().targetcornerlocations().size() == 8)
        {
            //The Target Corner List was used and is the right size... so just set the values
            //rather than deleting the list and starting over.
            for (int i = 0; i < 4; i++)
            {
                TargetInfoResultsPBMsg.mutable_imagelocation()->set_targetcornerlocations(2*i, imgMetaData->ImageCorners[i].x);
                TargetInfoResultsPBMsg.mutable_imagelocation()->set_targetcornerlocations(2*i + 1, imgMetaData->ImageCorners[i].y);
            }
        }
        else
        {
            //The Target Corner locations have not been previously setup...
            //Add to the list of corner locations here.
            TargetInfoResultsPBMsg.mutable_imagelocation()->clear_targetcornerlocations();
            for (int i = 0; i < 4; i++)
            {
                TargetInfoResultsPBMsg.mutable_imagelocation()->add_targetcornerlocations(imgMetaData->ImageCorners[i].x);
                TargetInfoResultsPBMsg.mutable_imagelocation()->add_targetcornerlocations(imgMetaData->ImageCorners[i].y);
            }
        }
    }

    bool ImageProcTargetInfoResultsMessage::AddTarget(int targetType,
                                                    double pixX, double pixY, double orientationRad,
                                                      LatLonAltCoord_t &latLonAlt,
                                                      AzimuthElevation_t &azimuthElevation)
    {
        bool error = false;
        try
        {
            GroundTargetLocationPBMsg *tgtLocMsg = new GroundTargetLocationPBMsg();
            tgtLocMsg->set_targettypecode(targetType);
            tgtLocMsg->set_targetpixellocation_x((int32_t) pixX);
            tgtLocMsg->set_targetpixellocation_y((int32_t) pixY);
            tgtLocMsg->set_targetorientationradians(orientationRad);
            tgtLocMsg->set_targetlatituderadians(latLonAlt.LatitudeRadians());
            tgtLocMsg->set_targetlongituderadians(latLonAlt.LongitudeRadians());
            tgtLocMsg->set_targetaltitudemsl(latLonAlt.Altitude());
            tgtLocMsg->set_targetazimuthradians(azimuthElevation.AzimuthAngleRad());
            tgtLocMsg->set_targetelevationradians(azimuthElevation.ElevationAngleRad());

            if (TargetInfoResultsPBMsg.mutable_targetlocations() == nullptr)
            {
                TargetInfoResultsPBMsg.add_targetlocations();
            }
            TargetInfoResultsPBMsg.mutable_targetlocations()->AddAllocated(tgtLocMsg);
        }
        catch (std::exception &e)
        {
            error = true;
            //LOGERROR("ImageCaptureManager: Exception: " << e.what());
        }
        return error;
    }

}