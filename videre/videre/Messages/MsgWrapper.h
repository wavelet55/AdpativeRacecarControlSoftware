/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Sept. 2020
 *
 * Message wrapper for transport over ZMQ or other serial interface.
  *******************************************************************/


#ifndef GROUNDFORCE360_DEV_MSGWRAPPER_H
#define GROUNDFORCE360_DEV_MSGWRAPPER_H

#include <string>
#include <RabitMessage.h>
#include <ByteArrayReaderWriterVidere.h>

namespace videre
{
    enum MsgSerializationType_e
    {
        MST_DtiByteArray,
        MST_Json,
        MST_ExerciseDataByteArray
    };

    const Rabit::EndianOrder_e MsgEndianOrder = Rabit::EndianOrder_e::Endian_Big;

    class MsgWrapper
    {
    private:
        uint8_t *_msgBuf;
        int _bufSize = 0;
        int _serializedMsgSize = 0;
        int _msgHeaderSize = 0;
        int _hdrMsgDataSizeIdx = 0;

    public:
        const int MAX_MSGHEADERSIZE = 256;
        std::string MsgName;
        std::string MsgQualifier;
        int32_t MsgValue;               //Generic Message Value... Depends on the Message.
        MsgSerializationType_e MsgSerializationType;
        uint32_t MsgDataSize;
        uint32_t MsgCRC;
        uint8_t *MsgData;

        bool FreeMsgDataWhenDone = false;


        MsgWrapper()
        {
            _msgBuf = nullptr;
            FreeMsgDataWhenDone = false;
            _bufSize = 0;
            clear();
        }

        MsgWrapper(uint8_t *msgBuf, int bufSize,
                   MsgSerializationType_e sType = MsgSerializationType_e::MST_DtiByteArray)
        {
            clear();
            _msgBuf = msgBuf;
            _bufSize = bufSize;
            MsgSerializationType = sType;
        }

        ~MsgWrapper()
        {
            if(FreeMsgDataWhenDone && MsgData != nullptr)
            {
                delete[] MsgData;
            }
            FreeMsgDataWhenDone = false;
            MsgData = nullptr;
        }

        void clear()
        {
            ReleaseMessageDataBuffer();
            _serializedMsgSize = 0;
            _msgHeaderSize = 0;
            _hdrMsgDataSizeIdx = 0;
            MsgName = "";
            MsgQualifier = "";
            MsgValue = 0;
            MsgSerializationType = MsgSerializationType_e::MST_DtiByteArray;
            MsgDataSize = 0;
            MsgCRC = 0;
            FreeMsgDataWhenDone = false;
            MsgData = nullptr;
        }

        void setFixedSizeBuffer(uint8_t *bufPtr, uint32_t bufSize)
        {
            ReleaseMessageDataBuffer();
            _msgBuf = bufPtr;
            _bufSize = bufSize;
            FreeMsgDataWhenDone = false;
        }

        //Allocate a Message Buffer.  It will be set
        //to be destroyed when done.
        //Returns true if error, false if buffer is allocated.
        bool AllocateMessageDataBuffer(int bufSize);

        void ReleaseMessageDataBuffer();

        int getSerializedMsgSize()
        {
            return _serializedMsgSize;
        }

        uint8_t *getMsgBuffer()
        {
            return _msgBuf;
        }

        //Pass in a buffer that has enough room to serialize the message.
        //The maxBuffersize indicates the maximum number of bytes available
        //in the buffer.
        //Returns the number of bytes used in the buffer (serialization size).
        int Serialize(Rabit::RabitMessage *msg = nullptr);

        int Serialize(std::shared_ptr<Rabit::RabitMessage> msg)
        {
            if(msg != nullptr)
            {
                return Serialize(msg.get());
            }
            else
            {
                return Serialize(nullptr);
            }
        }

        int Serialize(uint8_t *buf, int maxBufSize, std::shared_ptr<Rabit::RabitMessage> msg)
        {
            ReleaseMessageDataBuffer();
            _msgBuf = buf;
            _bufSize = maxBufSize;
            if(msg != nullptr)
            {
                return Serialize(msg.get());
            }
            else
            {
                return Serialize(nullptr);
            }
        }

        int Serialize(uint8_t *buf, int maxBufSize, Rabit::RabitMessage *msg = nullptr)
        {
            ReleaseMessageDataBuffer();
            _msgBuf = buf;
            _bufSize = maxBufSize;
            return Serialize(msg);
        }

        int Serialize(const std::string &strMsg);


        //Serialize the MsgWrapper Header.  This does not
        //fill-in the Message Data Size or Msg CRC.
        //Returns a pointer to the start of the data buffer.
        uint8_t * SerializeHeader();

        int FinishSerialization(int msgDataSize, uint32_t crc = 0, bool computeCRC = false );

        //The buffer contains the serialized message.
        //The len value indicates the length in bytes of the serialized data.
        //Returns a value indicating success.
        int DeSerializeHeader();

        //The buffer contains the serialized message.
        //The len value indicates the length in bytes of the serialized data.
        //Returns a value indicating success.
        int DeSerializeHeader(uint8_t *buf, int len)
        {
            FreeMsgDataWhenDone = false;
            _msgBuf = buf;
            _bufSize = len;
            return DeSerializeHeader();
        }


    };

}


#endif //GROUNDFORCE360_DEV_MSGWRAPPER_H
