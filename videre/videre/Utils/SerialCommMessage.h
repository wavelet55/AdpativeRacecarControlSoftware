/* ****************************************************************
 * Athr(s): Harry Direen PhD,
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: Jan. 13, 2018
 *
 * NeuroGroove IMU Communications Interface
 *******************************************************************/


#ifndef VIDERE_DEV_SERIALCOMMMESSAGE_H
#define VIDERE_DEV_SERIALCOMMMESSAGE_H

//#include <bits/shared_ptr.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace dtiUtils
{

    //A Message Char. Buffer.
    //Messages are null terminated to match the old c-style
    struct SerialCommMessage_t
    {
    private:
        int _maxMsgSize = 0;
        uint32_t _errorCount = 0;

    public:
        char* msgPtr = nullptr;
        int MsgSize = 0;
        int MsgId = 0;

        int getMaxMsgSize() {return _maxMsgSize;}

        SerialCommMessage_t() {}

        SerialCommMessage_t(int maxMsgSize)
        {
            init(maxMsgSize);
        }

        SerialCommMessage_t(const SerialCommMessage_t &scmsg);

        ~SerialCommMessage_t()
        {
            if(msgPtr != nullptr)
            {
                delete (msgPtr);
                msgPtr = nullptr;
            }
        }

        void init(int maxMsgSize);

        void releaseMem()
        {
            if(msgPtr != nullptr)
            {
                delete (msgPtr);
                msgPtr = nullptr;
                _maxMsgSize = 0;
            }
        }

        void clearMsg();

        void copyFrom(const SerialCommMessage_t &scmsg);

        void addMsg(std::string &msg);

        //Add noBytes to the message
        //starting from the buffer offset.
        //If msgOffset > 0 the bytes will be added to
        //the message starting at the given offset.
        void addMsg(u_char *cbuf, int noBytes,
                    int bufOffset = 0, int msgOffset = 0);

        //Get a copy of the message and return in the
        //cbuf... cbuf is expected to be large enough to fit the
        //message plus a null termination... so it should be atleast
        //MsgSize + 1.  The method returns the message size... not
        //including the null termination.
        int getMsg(u_char *cbuf);



    };

}
#endif //VIDERE_DEV_SERIALCOMMMESSAGE_H
