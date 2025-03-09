/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/

#include "RobotArm_RxMessageParser.h"
#include "Base64Conversions.h"

using namespace VidereUtils;

namespace videre
{

    RobotArm_RxMessageParser::RobotArm_RxMessageParser(Rabit::RabitManager* mgrPtr,
                                             std::shared_ptr<ConfigData> config)
    {
        _mgrPtr = mgrPtr;
        _config_sptr = config;
        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

    }

    void RobotArm_RxMessageParser::rxMsgHandler(dtiUtils::SerialCommMessage_t &msg)
    {
        int cmdRespBufIdx = 0;
        char* pCmd;
        char* pValue;

        int msgSize = msg.getMsg((u_char *)_receiveRS232CommMsgBuf);
        if(msgSize > 4 && _receiveRS232CommMsgBuf[3] == '=')
        {
            char* pCmd = _receiveRS232CommMsgBuf;
            char* pValue = _receiveRS232CommMsgBuf + 4;

            if(pCmd[0] == 'D')
            {

                std::cout << "RS232 Robot Arm Msg: " << pCmd << std::endl;
            }
            else
            {
                std::cout << "RS232 Robot Arm Msg: " << pCmd << std::endl;

            }






        }







    }


}