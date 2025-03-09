/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/

#include "SerialCommMessage.h"
#include <string>
#include <cstring>

namespace dtiUtils
{

    SerialCommMessage_t::SerialCommMessage_t(const SerialCommMessage_t &scmsg)
    {
        int i = 0;
        if(scmsg._maxMsgSize > 0)
        {
            msgPtr = new char[scmsg._maxMsgSize + 1];
            _maxMsgSize = scmsg._maxMsgSize;
            MsgSize = scmsg.MsgSize;
            std::memcpy(msgPtr, scmsg.msgPtr, MsgSize);
            msgPtr[MsgSize] = 0;
        }
    }

    void SerialCommMessage_t::init(int maxMsgSize)
    {
        //Add an extra char. to allow for null terminations.
        msgPtr = new char[maxMsgSize + 1];
        _maxMsgSize = maxMsgSize;
        clearMsg();
    }

    void SerialCommMessage_t::clearMsg()
    {
        if (_maxMsgSize > 0)
        {
            MsgSize = 0;
            msgPtr[0] = 0;
        }
    }

    void SerialCommMessage_t::copyFrom(const SerialCommMessage_t &scmsg)
    {
        int i = 0;
        if(scmsg.MsgSize > _maxMsgSize)
        {
            if(msgPtr != nullptr)
            {
                delete(msgPtr);
            }
            msgPtr = new char[scmsg._maxMsgSize + 1];
            _maxMsgSize = scmsg._maxMsgSize;
        }
        std::memcpy(msgPtr, scmsg.msgPtr, scmsg.MsgSize);
        msgPtr[scmsg.MsgSize] = 0;
        MsgSize = scmsg.MsgSize;
    }

    void SerialCommMessage_t::addMsg(std::string &msg)
    {
        int n = 0;
        if (_maxMsgSize > 0)
        {
            try
            {
                int max = msg.size() > _maxMsgSize ? _maxMsgSize : msg.size();
                std::memcpy(msgPtr, msg.c_str(), max);
                msgPtr[n] = max;
                MsgSize = max;
            }
            catch (std::exception &e)
            {
                ++_errorCount;
            }
        }
    }

    //Add noBytes to the message
    //starting from the buffer offset.
    //If msgOffset > 0 the bytes will be added to
    //the message starting at the given offset.
    void SerialCommMessage_t::addMsg(u_char *cbuf, int noBytes, int bufOffset, int msgOffset)
    {
        int n = msgOffset < 0 ? 0 : msgOffset > _maxMsgSize ? _maxMsgSize : msgOffset;
        if (_maxMsgSize > 0)
        {
            try
            {
                int availbleMsgSize = _maxMsgSize - n;
                if(availbleMsgSize > 0)
                {
                    int max = noBytes >= availbleMsgSize ? availbleMsgSize : noBytes;
                    std::memcpy(msgPtr + n, cbuf + bufOffset, max);
                    n += max;
                    msgPtr[n] = 0;
                    MsgSize = n;
                }
                else
                {
                    ++_errorCount;
                }
            }
            catch (std::exception &e)
            {
                ++_errorCount;
            }
        }
    }

    int SerialCommMessage_t::getMsg(u_char *cbuf)
    {
        int msgSize = 0;
        if(_maxMsgSize > 0)
        {
            std::memcpy(cbuf, msgPtr, MsgSize + 1);
            msgSize = MsgSize;
        }
        return msgSize;
    }

}
