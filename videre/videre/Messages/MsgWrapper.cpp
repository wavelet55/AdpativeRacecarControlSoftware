/* ****************************************************************
 * Message
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Sept. 2020
 *
 * Message wrapper for transport over ZMQ or other serial interface.
  *******************************************************************/


#include "MsgWrapper.h"
#include "ByteArrayReaderWriter.h"
#include "CRC_Calculator.h"
#include <new>

using namespace Rabit;

namespace videre
{

    //Allocate a Message Buffer.  It will be set
    //to be destroyed when done.
    //Returns true if error, false if buffer is allocated.
    bool MsgWrapper::AllocateMessageDataBuffer(int bufSize)
    {
        bool error = true;
        //Release previous buffer
        ReleaseMessageDataBuffer();
        try
        {
            //Add a max serialized message header size to the buffer.
            _msgBuf = new uint8_t[bufSize + MAX_MSGHEADERSIZE];
        }
        catch (std::exception &e)
        {
            //LOGERROR("MsgWrapper:AllocateMessageDataBuffer Exception: " << e.what());
            error = true;
            _msgBuf = nullptr;
        }
        if(_msgBuf != nullptr)
        {
            _bufSize = (uint32_t)bufSize + MAX_MSGHEADERSIZE;
            error = false;
        }
        return error;
    }

    void MsgWrapper::ReleaseMessageDataBuffer()
    {
        if(FreeMsgDataWhenDone && _msgBuf != nullptr)
        {
            delete[] _msgBuf;
        }
        FreeMsgDataWhenDone = false;
        _msgBuf = nullptr;
        _bufSize = 0;
    }


    //The "stype" may be used by the end-user to indicate the
    //type of serialization to perform.
    //Pass in a buffer that has enough room to serialize the message.
    //The maxBuffersize indicates the maximum number of bytes available
    //in the buffer.
    //Returns the number of bytes used in the buffer (serialization size).
    int MsgWrapper::Serialize(Rabit::RabitMessage *msg)
    {
        _serializedMsgSize = 0;
        if(_msgBuf != nullptr && _bufSize > 0)
        {
            //There is a bit of a game in the serialization process.  We
            //do not know the size of the message header due to the Message Wrapper Names.
            //We also do not know the serialized size of the rabit message until we
            //serialize it.  So there is some back and forth to get all the info
            //correct.
            ByteArrayWriter bw(_msgBuf, _bufSize, MsgEndianOrder);
            bw.writeString(MsgName);
            bw.writeString(MsgQualifier);
            bw.writeInt32(MsgValue);
            bw.writeByte((uint8_t) MsgSerializationType);
            _hdrMsgDataSizeIdx = bw.Idx;
            _msgHeaderSize = _hdrMsgDataSizeIdx + 2 * sizeof(uint32_t);

            if (msg != nullptr) {
                bw.Idx = _msgHeaderSize;
                MsgData = bw.GetByteArrayAddrPtrAtIdx();
                MsgDataSize = msg->Serialize(MsgData,
                                         bw.GetNumberBytesAvailable(),
                                         (int) MsgSerializationType);

                bw.Idx = _msgHeaderSize;
                MsgCRC = Compute_CRC16((char *) bw.GetByteArrayAddrPtrAtIdx(), MsgDataSize);
            }
            else if(MsgData != nullptr && MsgDataSize > 0)
            {
                bw.Idx = _msgHeaderSize;
                bw.WriteBytes(MsgData, MsgDataSize, 0, MsgDataSize);
                bw.Idx = _msgHeaderSize;
                MsgCRC = Compute_CRC16((char *) bw.GetByteArrayAddrPtrAtIdx(), MsgDataSize);
            }
            else
            {
                MsgDataSize = 0;
                MsgCRC = 0;
            }

            bw.Idx = _hdrMsgDataSizeIdx;
            bw.writeUInt32(MsgDataSize);
            bw.writeUInt32(MsgCRC);
        }
        _serializedMsgSize = MsgDataSize + _msgHeaderSize;
        return _serializedMsgSize;
    }

    //The "stype" may be used by the end-user to indicate the
    //type of serialization to perform.
    //Pass in a buffer that has enough room to serialize the message.
    //The maxBuffersize indicates the maximum number of bytes available
    //in the buffer.
    //Returns the number of bytes used in the buffer (serialization size).
    int MsgWrapper::Serialize(const std::string &strMsg)
    {
        _serializedMsgSize = 0;
        if(_msgBuf != nullptr && _bufSize > 0)
        {
            //There is a bit of a game in the serialization process.  We
            //do not know the size of the message header due to the Message Wrapper Names.
            //We also do not know the serialized size of the rabit message until we
            //serialize it.  So there is some back and forth to get all the info
            //correct.
            ByteArrayWriter bw(_msgBuf, _bufSize, MsgEndianOrder);
            bw.writeString(MsgName);
            bw.writeString(MsgQualifier);
            bw.writeInt32(MsgValue);
            bw.writeByte((uint8_t) MsgSerializationType);
            _hdrMsgDataSizeIdx = bw.Idx;
            _msgHeaderSize = _hdrMsgDataSizeIdx + 2 * sizeof(uint32_t);

            if (!strMsg.empty())
            {
                for(int n = 0; n < strMsg.length(); n++)
                {
                    bw.writeChar(strMsg[n]);
                }
                MsgDataSize = strMsg.length();
                bw.Idx = _msgHeaderSize;
                MsgCRC = Compute_CRC16((char *) bw.GetByteArrayAddrPtrAtIdx(), MsgDataSize);
            }
            else
            {
                MsgDataSize = 0;
                MsgCRC = 0;
            }

            bw.Idx = _hdrMsgDataSizeIdx;
            bw.writeUInt32(MsgDataSize);
            bw.writeUInt32(MsgCRC);
        }
        _serializedMsgSize = MsgDataSize + _msgHeaderSize;
        return _serializedMsgSize;
    }

    //Serialize the MsgWrapper Header.  This does not
    //fill-in the Message Data Size or Msg CRC.
    //Returns a pointer to the start of the data buffer.
    uint8_t * MsgWrapper::SerializeHeader()
    {
        _serializedMsgSize = 0;
        MsgData = nullptr;
        if(_msgBuf != nullptr && _bufSize > 0)
        {
            //There is a bit of a game in the serialization process.  We
            //do not know the size of the message header due to the Message Wrapper Names.
            //We also do not know the serialized size of the rabit message until we
            //serialize it.  So there is some back and forth to get all the info
            //correct.
            ByteArrayWriter bw(_msgBuf, _bufSize, MsgEndianOrder);
            bw.writeString(MsgName);
            bw.writeString(MsgQualifier);
            bw.writeInt32(MsgValue);
            bw.writeByte((uint8_t) MsgSerializationType);
            _hdrMsgDataSizeIdx = bw.Idx;
            //Write default values in the MsgDataSize and CRC
            bw.writeUInt32(0);
            bw.writeUInt32(0);
            _msgHeaderSize = bw.Idx;
            MsgData = MsgData = bw.GetByteArrayAddrPtrAtIdx();
        }
        return MsgData;
    }

    int MsgWrapper::FinishSerialization(int msgDataSize, uint32_t crc, bool computeCRC )
    {
        MsgDataSize = msgDataSize;
        MsgCRC = crc;
        if(computeCRC && MsgData != nullptr)
        {
            MsgCRC = Compute_CRC16((char *)MsgData, MsgDataSize);
        }
        ByteArrayWriter bw(_msgBuf, _bufSize, MsgEndianOrder);
        bw.Idx = _hdrMsgDataSizeIdx;
        bw.writeUInt32(MsgDataSize);
        bw.writeUInt32(MsgCRC);
        _serializedMsgSize = MsgDataSize + _msgHeaderSize;
        return _serializedMsgSize;
    }

    //The "stype" may be used by the end-user to indicate the
    //type of serialization to perform.
    //The buffer contains the serialized message.
    //The len value indicates the length in bytes of the serialized data.
    //Returns a value indicating success.
    int MsgWrapper::DeSerializeHeader()
    {
        int msgSize = 0;
        if(_msgBuf != nullptr && _bufSize > 0)
        {
            ByteArrayReader br(_msgBuf, _bufSize, MsgEndianOrder);
            br.readString(&MsgName);
            br.readString(&MsgQualifier);
            MsgValue = br.readInt32();
            MsgSerializationType = (MsgSerializationType_e) br.readByte();
            MsgDataSize = br.readUInt32();
            MsgCRC = br.readUInt32();
            if(MsgDataSize > 0)
            {
                MsgData = &br.ByteArray[br.Idx];
            }
            else
            {
                MsgData = nullptr;
            }
            msgSize = br.Idx;
        }
        return msgSize;
    }



}