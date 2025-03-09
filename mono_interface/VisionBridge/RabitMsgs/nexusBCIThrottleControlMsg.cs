
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file nexusBCIThrottleControlMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class nexusBCIThrottleControlMsg : VSMessage
    {

        public bool ThrottleOn { get; set;}

       
        public nexusBCIThrottleControlMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            ThrottleOn = false;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeBool(ThrottleOn);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            ThrottleOn = br.readBool();

            return br.Idx;
        }
    }
}
