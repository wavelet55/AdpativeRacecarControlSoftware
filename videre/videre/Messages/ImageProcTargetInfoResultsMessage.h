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

#ifndef VIDERE_DEV_IMAGEPROCTARGETINFORESULTSMESSAGE_H
#define VIDERE_DEV_IMAGEPROCTARGETINFORESULTSMESSAGE_H

#include <iostream>
#include <string>
#include <RabitMessage.h>
#include "XYZCoord_t.h"
#include "LatLonAltStruct.h"
#include "RollPitchYaw_t.h"
#include "AzimuthElevation_t.h"
//#include "VehicleInertialStatesMessage.h"
//#include "CameraOrientationMessage.h"
#include "image_plus_metadata_message.h"
#include "../ProtobufMessages/vision_messages.pb.h"

using namespace std;
using namespace GeoCoordinateSystemNS;
using namespace vision_messages;

namespace videre
{
    //This is a container message for the Protobuf message:
    //ImageProcTargetInfoResultsPBMsg
    //The container or wrapper makes it easier to use the ImageProcTargetInfoResultsPBMsg
    //and allows the message to work within the rest of the Vision System.
    class ImageProcTargetInfoResultsMessage : public Rabit::RabitMessage
    {
    private:


    public:
        ImageProcTargetInfoResultsPBMsg  TargetInfoResultsPBMsg;



    public:

        ImageProcTargetInfoResultsMessage() : RabitMessage()
        {
            Clear();
        }

        ImageProcTargetInfoResultsMessage(const ImageProcTargetInfoResultsMessage& msg)
        {
            *this = msg;
        }

        void Clear();

        virtual unique_ptr<Rabit::RabitMessage> Clone() const final;

        virtual bool CopyMessage(Rabit::RabitMessage *msg) final;


        void SetImageMetaDataInfo(const ImagePlusMetadataMessage *imgMetaData);

        bool AddTarget(int targetType,
                       double pixX, double pixY, double orientationRad,
                       LatLonAltCoord_t &latLonAlt,
                        AzimuthElevation_t &azimuthElevation);

    };

}



#endif //VIDERE_DEV_IMAGEPROCTARGETINFORESULTSMESSAGE_H
