/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Aug. 2016
 *
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *******************************************************************/

#include "CommUtils.h"
using namespace std;

namespace videre
{

    void ImageCaptureStatusMsgToProtobufMsg(std::shared_ptr<ImageCaptureStatusMessage> mgrStatsMsg,
                                            vision_messages::ImageCaptureStatusPBMsg &imgCapStatsPBMsg)
    {
        imgCapStatsPBMsg.set_imagecaptureenabled(mgrStatsMsg->ImageCaptureEnabled);
        imgCapStatsPBMsg.set_imagecapturecomplete(mgrStatsMsg->ImageCaptureComplete);
        imgCapStatsPBMsg.set_endofimages(mgrStatsMsg->EndOfImages);
        imgCapStatsPBMsg.set_totalnumberofimagescaptured(mgrStatsMsg->TotalNumberOfImagesCaptured);
        imgCapStatsPBMsg.set_currentnumberofimagescaptured(mgrStatsMsg->CurrentNumberOfImagesCaptured);
        imgCapStatsPBMsg.set_averageframespersecond(mgrStatsMsg->AverageFramesPerSecond);
        imgCapStatsPBMsg.set_imagecapturesource((vision_messages::ImageCaptureSource_e)mgrStatsMsg->ImageCaptureSource);
        imgCapStatsPBMsg.set_errorcode((vision_messages::ImageCaptureError_e)mgrStatsMsg->ErrorCode);
    }

    void ManagerStatsMsgToProtobufMsg(std::shared_ptr<Rabit::ManagerStatusMessage> mgrStatsMsg,
                                      vision_messages::ManagerStatsPBMsg &mgrStatsTxPBMsg)
    {
        //Translate message to the Protobuf Message
        mgrStatsTxPBMsg.set_managername(mgrStatsMsg->ManagerName);
        mgrStatsTxPBMsg.set_runningstate((int32_t) mgrStatsMsg->RunningState);
        mgrStatsTxPBMsg.set_errorcondition(mgrStatsMsg->ErrorCondition);
        mgrStatsTxPBMsg.set_errorcode(mgrStatsMsg->ErrorCode);

        mgrStatsTxPBMsg.set_totalnumberofexecutecycles(mgrStatsMsg->ManagerStats.TotalNumberOfExecuteCycles);
        mgrStatsTxPBMsg.set_numberofexecutecycles(mgrStatsMsg->ManagerStats.NumberOfExecuteCycles);
        mgrStatsTxPBMsg.set_timesincelaststatsreset_sec(mgrStatsMsg->ManagerStats.TimeSinceLastStatsReset_Sec);
        mgrStatsTxPBMsg.set_minexecuteunitofworktime_sec(mgrStatsMsg->ManagerStats.MinExecuteUnitOfWorkTime_Sec);
        mgrStatsTxPBMsg.set_maxexecuteunitofworktime_sec(mgrStatsMsg->ManagerStats.MaxExecuteUnitOfWorkTime_Sec);
        mgrStatsTxPBMsg.set_aveexecuteunitofworktime_sec(mgrStatsMsg->ManagerStats.AveExecuteUnitOfWorkTime_Sec);
        mgrStatsTxPBMsg.set_totalexecuteunitofworktime_sec(mgrStatsMsg->ManagerStats.TotalExecuteUnitOfWorkTime_Sec);

        mgrStatsTxPBMsg.set_minsleeptime_sec(mgrStatsMsg->ManagerStats.MinSleepTime_Sec);
        mgrStatsTxPBMsg.set_maxsleeptime_sec(mgrStatsMsg->ManagerStats.MaxSleepTime_Sec);
        mgrStatsTxPBMsg.set_avesleeptime_sec(mgrStatsMsg->ManagerStats.AveSleepTime_Sec);
        mgrStatsTxPBMsg.set_totalsleeptime_sec(mgrStatsMsg->ManagerStats.TotalSleepTime_Sec);

        mgrStatsTxPBMsg.set_numberofwakeupcallswhileasleep(mgrStatsMsg->ManagerStats.NumberOfWakeUpCallsWhileAsleep);
        mgrStatsTxPBMsg.set_numberofwakeupcallswhileawake(mgrStatsMsg->ManagerStats.NumberOfWakeUpCallsWhileAwake);
    }

    bool CompressedImgageMsgToTxMsg(CompressedImageMessage *cImgMsgPtr,
                                    vision_messages::VisionMessageWrapperPBMsg &msgWrapperPBMsg)
    {
        bool error = true;
        std::ostringstream imgInfo;
        try
        {
            msgWrapperPBMsg.set_msgname("CompressedImageMsg");
            imgInfo << "ImageFormat=" << cImgMsgPtr->ImageFormatAsString()
                      << ",ImageNumber=" << cImgMsgPtr->ImageNumber;
            msgWrapperPBMsg.set_msgqualifier(imgInfo.str());
            msgWrapperPBMsg.set_msgdatasize(cImgMsgPtr->ImageBuffer.size());
            msgWrapperPBMsg.set_msgdata(cImgMsgPtr->ImageBuffer.data(), cImgMsgPtr->ImageBuffer.size());
            error = false;
        }
        catch (std::exception &e)
        {
             error = true;
        }
        return error;
    }

    bool TargetInfoMsgToTxMsg(ImageProcTargetInfoResultsMessage *tgtMsgPtr,
                                    vision_messages::VisionMessageWrapperPBMsg &msgWrapperPBMsg)
    {
        bool error = true;
        try
        {
            msgWrapperPBMsg.set_msgname("TargetInformationMsg");
            msgWrapperPBMsg.set_msgdata(tgtMsgPtr->TargetInfoResultsPBMsg.SerializeAsString());
            error = false;
        }
        catch (std::exception &e)
        {
            error = true;
        }
        return error;
    }



    bool ManagerStatsMsgToTxMsg(std::shared_ptr<Rabit::ManagerStatusMessage> mgrStatsMsg,
                                vision_messages::VisionMessageWrapperPBMsg &msgWrapperPBMsg)
    {
        bool error = true;
        vision_messages::ManagerStatsPBMsg mgrStatsTxPBMsg;
        try
        {
            ManagerStatsMsgToProtobufMsg(mgrStatsMsg, mgrStatsTxPBMsg);

            msgWrapperPBMsg.Clear();
            msgWrapperPBMsg.set_msgname("ManagerStatsMsg");
            msgWrapperPBMsg.set_msgqualifier(mgrStatsMsg->ManagerName);
            msgWrapperPBMsg.set_msgdata(mgrStatsTxPBMsg.SerializeAsString());
            error = false;
        }
        catch (std::exception &e)
        {
            error = true;
        }
        return error;
    }

    bool VehicleInertialStatesPBMsgToVISMsg(VehicleInertialStatePBMsg &visPBMsg,
                                            std::shared_ptr<VehicleInertialStatesMessage> visMsg_sptr)
    {
        bool error = true;
        try
        {
            if(visPBMsg.coordinateslatlonorxy())
            {
                LatLonAltCoord_t llaPos(visPBMsg.latituderadory(),
                                        visPBMsg.longituderadorx(),
                                        visPBMsg.altitudemsl(), false);
                visMsg_sptr->SetLatLonAltConvertToXYZ(llaPos);
            }
            else
            {
                XYZCoord_t xyzPos(visPBMsg.longituderadorx(),
                                  visPBMsg.latituderadory(),
                                  visPBMsg.altitudemsl());

                visMsg_sptr->SetXYZCoordinatesConverToLatLonAlt(xyzPos);
            }

            visMsg_sptr->HeightAGL = visPBMsg.heightagl();

            visMsg_sptr->XYZVelocities.x = visPBMsg.veleastmps();
            visMsg_sptr->XYZVelocities.y = visPBMsg.velnorthmps();
            visMsg_sptr->XYZVelocities.z = visPBMsg.veldownmps();

            visMsg_sptr->RollPitchYaw.IsRate = false;
            visMsg_sptr->RollPitchYaw.SetRollRadians(visPBMsg.rollrad());
            visMsg_sptr->RollPitchYaw.SetPitchRadians(visPBMsg.pitchrad());
            visMsg_sptr->RollPitchYaw.SetYawRadians(visPBMsg.yawrad());

            visMsg_sptr->RollPitchYawRates.IsRate = true;
            visMsg_sptr->RollPitchYawRates.SetRollRadians(visPBMsg.rollrateradps());
            visMsg_sptr->RollPitchYawRates.SetPitchRadians(visPBMsg.pitchrateradps());
            visMsg_sptr->RollPitchYawRates.SetYawRadians(visPBMsg.yawrateradps());

            visMsg_sptr->GpsTimeStampSec = visPBMsg.gpstimestampsec();
            error = false;
        }
        catch (std::exception &e)
        {
            error = true;
        }
        return error;
    }

    bool CameraSteeringPBMsgToCOMsg(CameraSteeringPBMsg &coPBMsg,
                                            std::shared_ptr<CameraOrientationMessage> coMsg_sptr)
    {
        bool error = true;
        try
        {
            coMsg_sptr->CameraSteeringModeSPOI = coPBMsg.camerasteeringmodespoi();
            if (coPBMsg.camerasteeringmodespoi())
            {
                if (coPBMsg.coordinateslatlonorxy())
                {
                    LatLonAltCoord_t llaPos(coPBMsg.spoilatituderadory(),
                                            coPBMsg.spoilongituderadorx(),
                                            0, false);
                    coMsg_sptr->SetSPOILatLonConvertToXY(llaPos);
                }
                else
                {
                    XYZCoord_t xyzPos(coPBMsg.spoilongituderadorx(),
                                      coPBMsg.spoilatituderadory(),
                                      0);

                    coMsg_sptr->SetSPOIXYCoordinatesConverToLatLon(xyzPos);
                }
            }
            else
            {
                coMsg_sptr->CameraAzimuthElevationAngles.SetAzimuthAngleRad(
                        coPBMsg.cameraazimuthanglerad());
                coMsg_sptr->CameraAzimuthElevationAngles.SetElevationAngleRad(
                        coPBMsg.cameraelevationanglerad());
            }

            coMsg_sptr->SetCameraZoomPercent(coPBMsg.camerazoompercent());

            error = false;
        }
        catch (std::exception &e)
        {
            error = true;
        }
        return error;
    }

    bool IMUAccelGyroMsgToTxMsg(AccelerometerGyroMessage * agMsgPtr,
                                vision_messages::VisionMessageWrapperPBMsg &msgWrapperPBMsg)
    {
        bool error = true;
        try
        {
            msgWrapperPBMsg.set_msgname("IMUAccelGyroDataMsg");

            vision_messages::IMUAccelGyroDataPBMsg imuPBmsg;
            imuPBmsg.set_imulocation((int)agMsgPtr->IMU_SensorID);
            imuPBmsg.set_imutimestampsec(agMsgPtr->IMUTimeStampSec);
            imuPBmsg.set_videretimestampsec(agMsgPtr->TimeStampSeconds());
            imuPBmsg.set_accelmps2_x(agMsgPtr->AccelerationRates.x);
            imuPBmsg.set_accelmps2_y(agMsgPtr->AccelerationRates.y);
            imuPBmsg.set_accelmps2_z(agMsgPtr->AccelerationRates.z);
            imuPBmsg.set_gyroradpersec_x(agMsgPtr->GyroAngularRates.x);
            imuPBmsg.set_gyroradpersec_y(agMsgPtr->GyroAngularRates.y);
            imuPBmsg.set_gyroradpersec_z(agMsgPtr->GyroAngularRates.z);
            msgWrapperPBMsg.set_msgdata(imuPBmsg.SerializeAsString());
            error = false;
        }
        catch (std::exception &e)
        {
            error = true;
        }
        return error;
    }

    bool TrackHeadOrientationMsgToTxMsg(TrackHeadOrientationMessage * htoMsgPtr,
                                vision_messages::VisionMessageWrapperPBMsg &msgWrapperPBMsg)
    {
        bool error = true;
        try
        {
            msgWrapperPBMsg.set_msgname("TrackHeadOrientationMsg");

            vision_messages::TrackHeadOrientationPBMsg thoPBMsg;
            thoPBMsg.set_headorientationquaternion_w(htoMsgPtr->TrackHeadOrientationData.HeadOrientationQuaternion.qScale);
            thoPBMsg.set_headorientationquaternion_x(htoMsgPtr->TrackHeadOrientationData.HeadOrientationQuaternion.qVec.x);
            thoPBMsg.set_headorientationquaternion_y(htoMsgPtr->TrackHeadOrientationData.HeadOrientationQuaternion.qVec.y);
            thoPBMsg.set_headorientationquaternion_z(htoMsgPtr->TrackHeadOrientationData.HeadOrientationQuaternion.qVec.z);

            thoPBMsg.set_headtranslationvec_x(htoMsgPtr->TrackHeadOrientationData.HeadTranslationVec.x);
            thoPBMsg.set_headtranslationvec_y(htoMsgPtr->TrackHeadOrientationData.HeadTranslationVec.y);
            thoPBMsg.set_headtranslationvec_z(htoMsgPtr->TrackHeadOrientationData.HeadTranslationVec.z);
            thoPBMsg.set_isdatavalid(htoMsgPtr->TrackHeadOrientationData.IsDataValid);
            thoPBMsg.set_imagenumber(htoMsgPtr->ImageNumber);
            thoPBMsg.set_imagecapturetimestampsec(htoMsgPtr->ImageCaptureTimeStampSec);
            thoPBMsg.set_videretimestampsec(htoMsgPtr->TimeStampSeconds());
            thoPBMsg.set_covariancenorm(htoMsgPtr->CovarianceNorm);
            msgWrapperPBMsg.set_msgdata(thoPBMsg.SerializeAsString());
            error = false;
        }
        catch (std::exception &e)
        {
            error = true;
        }
        return error;
    }

    bool GPSFixeMsgToTxMsg(GPSFixMessage * gpsMsgPtr,
                                        vision_messages::VisionMessageWrapperPBMsg &msgWrapperPBMsg)
    {
        bool error = true;
        try
        {
            msgWrapperPBMsg.set_msgname("GPSFixMsg");

            vision_messages::GPSFixPBMsg gpsPBMsg;
            gpsPBMsg.set_trackingsatellites(gpsMsgPtr->GPSFixData.trackingSatellites);
            gpsPBMsg.set_gpstimestampsec(gpsMsgPtr->GPSFixData.timestamp.rawTime);
            gpsPBMsg.set_videretimestampsec(gpsMsgPtr->TimeStampSeconds());
            gpsPBMsg.set_latitudedeg(gpsMsgPtr->GPSFixData.latitude);
            gpsPBMsg.set_longitudedeg(gpsMsgPtr->GPSFixData.longitude);
            gpsPBMsg.set_altitudemsl(gpsMsgPtr->GPSFixData.altitude);
            gpsPBMsg.set_position_x(gpsMsgPtr->XYZPositionMeters.x);
            gpsPBMsg.set_position_y(gpsMsgPtr->XYZPositionMeters.y);
            gpsPBMsg.set_position_z(gpsMsgPtr->XYZPositionMeters.z);
            gpsPBMsg.set_velocity_x(gpsMsgPtr->XYZVelocityMetersPerSec.x);
            gpsPBMsg.set_velocity_y(gpsMsgPtr->XYZVelocityMetersPerSec.y);
            gpsPBMsg.set_velocity_z(gpsMsgPtr->XYZVelocityMetersPerSec.z);

            msgWrapperPBMsg.set_msgdata(gpsPBMsg.SerializeAsString());
            error = false;
        }
        catch (std::exception &e)
        {
            error = true;
        }
        return error;
    }

}