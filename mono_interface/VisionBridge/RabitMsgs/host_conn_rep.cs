
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file host_conn_rep.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class host_conn_rep : VSMessage
    {

        public UInt64 timestamp { get; set;}
        public UInt8 sys_id { get; set;}
        public UInt8 seq { get; set;}
        public Int64 tc1 { get; set;}
        public Int64 ts1 { get; set;}

       
        public host_conn_rep()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            timestamp = 0;
            sys_id = 0;
            seq = 0;
            tc1 = 0;
            ts1 = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeUInt64(timestamp);
            bw.writeUInt8(sys_id);
            bw.writeUInt8(seq);
            bw.writeInt64(tc1);
            bw.writeInt64(ts1);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            timestamp = br.readUInt64();
            sys_id = br.readUInt8();
            seq = br.readUInt8();
            tc1 = br.readInt64();
            ts1 = br.readInt64();

            return br.Idx;
        }
    }
}
