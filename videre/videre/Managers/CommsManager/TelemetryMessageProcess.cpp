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

#include "TelemetryMessageProcess.h"
#include <global_defines.h>
#include "GeoCoordinateSystem.h"
#include "LatLonAltStruct.h"
#include "XYZCoord_t.h"

using namespace GeoCoordinateSystemNS;
using namespace MathLibsNS;

namespace videre
{

    TelemetryMessageProcessor::TelemetryMessageProcessor()
    {
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
    }

    bool TelemetryMessageProcessor::Intialize(Rabit::RabitManager* mgrPtr, ZeroMQWrapper* zmqComm)
    {
        bool error = false;
        _mgrPtr = mgrPtr;
        _zmqCommPtr = zmqComm;

        //Messages
        _vehicleInertialStatesMsg_sptr = std::make_shared<VehicleInertialStatesMessage>();
        mgrPtr->AddPublishSubscribeMessage("VehicleInertialStatesMessage", _vehicleInertialStatesMsg_sptr);

        _cameraOrientationMsg_sptr = std::make_shared<CameraOrientationMessage>();
        mgrPtr->AddPublishSubscribeMessage("CameraOrientationMessage", _cameraOrientationMsg_sptr);

        //Create a Command Map
        _message_map = std::unordered_map<std::string, messageHandler_t>();

        //Add Message Handlers to the Dictionary.
        _message_map["VehicleInertialStates"] = boost::bind(&TelemetryMessageProcessor::VehicleInertialStatesHandler, this, _1);
        _message_map["CameraOrientation"] = boost::bind(&TelemetryMessageProcessor::CameraOrientationHandler, this, _1);

        return error;
    }

    void TelemetryMessageProcessor::Shutdown()
    {

    }

    //Call this method to check for and handle all messages from the remote system.
    //returns the number of messages processed.
    int TelemetryMessageProcessor::HandleMessagesFromRemote()
    {
        int noMsgsHandled = 0;
        bool messageProcessed = false;
        static zmq::message_t sub;

        while( _zmqCommPtr->SubscribedTelemetryFromHOPS(&sub))
        {
            _messageWrapper.Clear();
            //LOGTRACE("Received message from HOPS Framework.")
            if(_messageWrapper.ParseFromArray(sub.data(), sub.size()) )
            {
                DispatchToFunctionHandler(&_messageWrapper);
            }
            else
            {
                LOGERROR("TelemetryMessageProcessor:Error Deserializeing a command.");
            }
            ++noMsgsHandled;
        }
        return noMsgsHandled;
    }


    bool TelemetryMessageProcessor::DispatchToFunctionHandler(VisionMessageWrapperPBMsg *msgWpr)
    {
        bool is_valid_command = false;

        try
        {
            std::string msgTypeName = msgWpr->msgname();
            auto key_location = _message_map.find(msgTypeName);
            if (key_location != _message_map.end())
            {
                _message_map[msgTypeName](msgWpr);
                is_valid_command = true;
            }
            else
            {
                LOGWARN("TelemetryMessageProcessor:Invalid Message Type: " + msgTypeName);
                is_valid_command = false;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("TelemetryMessageProcessor:Exception in handler processing: " << e.what());
        }
        return is_valid_command;
    }



    void TelemetryMessageProcessor::VehicleInertialStatesHandler(VisionMessageWrapperPBMsg *msg)
    {
        if( msg->has_msgdata())
        {
            try
            {
                _vehicleInertialStatesPBMsg.ParseFromString(msg->msgdata());
                VehicleInertialStatesPBMsgToVISMsg(_vehicleInertialStatesPBMsg, _vehicleInertialStatesMsg_sptr);
                _vehicleInertialStatesMsg_sptr->PostMessage();
            }
            catch (std::exception &e)
            {
                LOGERROR("VehicleInertialStatesHandler Exception: " << e.what());
            }
        }

    }

    void TelemetryMessageProcessor::CameraOrientationHandler(VisionMessageWrapperPBMsg *msg)
    {
        if( msg->has_msgdata())
        {
            try
            {
                _cameraSteeringPBMsg.ParseFromString(msg->msgdata());

                _cameraOrientationMsg_sptr->Clear();

                CameraSteeringPBMsgToCOMsg(_cameraSteeringPBMsg, _cameraOrientationMsg_sptr);

                _cameraOrientationMsg_sptr->PostMessage();
            }
            catch (std::exception &e)
            {
                LOGERROR("CameraOrientationHandler Exception: " << e.what());
            }
        }
    }


}