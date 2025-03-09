/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: July, 2018
 *
 *******************************************************************/

#include "RemoteControlRxMessageParser.h"
#include "Base64Conversions.h"

using namespace VidereUtils;

namespace videre
{

    RemoteControlRxMessageParser::RemoteControlRxMessageParser(Rabit::RabitManager* mgrPtr,
                                             std::shared_ptr<ConfigData> config)
    {
        _mgrPtr = mgrPtr;
        _config_sptr = config;
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        _remoteControlInputMsg = std::make_shared<RemoteControlInputMessage>();
        _mgrPtr->AddPublishSubscribeMessage("RemoteControlInputMsg", _remoteControlInputMsg);


    }


    bool RemoteControlRxMessageParser::Initialize()
    {
        ReverseThrottleBrake = _config_sptr->GetConfigBoolValue("RemoteControl.ReverseThrottleBrake",false);
        ThrottleBrakeDeadBandPercent = _config_sptr->GetConfigDoubleValue("RemoteControl.ThrottleBrakeDeadBandPercent",5.0);
        ReverseSteeringControl = _config_sptr->GetConfigBoolValue("RemoteControl.ReverseSteeringControl",false);
        SteeringDeadbandPercent = _config_sptr->GetConfigDoubleValue("RemoteControl.SteeringDeadband",2.5);
    }


    int strToInt(char *cPtr)
    {
        int val = 0;
        int n = 0;
        for(int i = 0; i < 8; i++)
        {
            char c = cPtr[i];
            if(c >= '0' && c <= '9')
            {
                val = 10 * val + (int)(c - '0');
            }
            else if( c != ' ')
            {
                break;
            }
        }
        return val;
    }

    void RemoteControlRxMessageParser::rxMsgHandler(dtiUtils::SerialCommMessage_t &msg)
    {
        int cmdRespBufIdx = 0;
        char* pCmd = _receiveRS232CommMsgBuf;
        char* pValue;
        int pos;

        //Six Channels of values in the range of 1000 to 2000
        int values[6];
        bool dataOk = true;

        int msgSize = msg.getMsg((u_char *)_receiveRS232CommMsgBuf);
        if(msgSize > 30 )
        {
            //Skip the timestamp... we do not need it.
            pValue = strchr(pCmd, ',');
            for(int i = 0; i < 6; i++)
            {
                if(pValue != nullptr && (pValue - pCmd) < msgSize -2 )
                {
                    ++pValue;
                    values[i] = strToInt(pValue);
                    if(values[i] < 900 || values[i] > 2100)
                    {
                        //Invalid value found... trash the hole line
                        LOGWARN("RemoteControlRxMessageParser, Invalid Data: " << _receiveRS232CommMsgBuf);
                        dataOk = false;
                        break;
                    }
                }
                else
                {
                    dataOk = false;
                    break;
                }
                pValue = strchr(pValue, ',');
            }
            if(dataOk)
            {
                double val = 0.2 * (double)(values[0] - 1500);
                val = fabs(val) < SteeringDeadbandPercent ? 0.0 : val;
                val = ReverseSteeringControl ? -val : val;
                _remoteControlInputMsg->setSteeringControlPercent(val);
                //std::cout << "Steering Val: " << val << std::endl;

                val = 0.2 * (double)(values[1] - 1500);
                val = fabs(val) < ThrottleBrakeDeadBandPercent ? 0.0 : val;
                val = ReverseThrottleBrake ? -val : val;
                _remoteControlInputMsg->setThrottleBrakePercent(val);
                //std::cout << "Throttle Brake Val: " << val << std::endl;

                for(int i = 0; i < 4; i++)
                {
                    val = 0.2 * (double)(values[i + 2] - 1500);
                    _remoteControlInputMsg->setChannelNPercent(i, val);
                }
                _remoteControlInputMsg->PostMessage();
            }
        }
    }


}