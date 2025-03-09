
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file MgrDtxConnectionStatusMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class MgrDtxConnectionStatusMsg : VSMessage
    {

        public UInt8 dtx_connection_status { get; set;}
        public Int32 mgr_status { get; set;}

       
        public MgrDtxConnectionStatusMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            dtx_connection_status = 0;
            mgr_status = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeUInt8(dtx_connection_status);
            bw.writeInt32(mgr_status);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            dtx_connection_status = br.readUInt8();
            mgr_status = br.readInt32();

            return br.Idx;
        }
    }
}
