
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file sipnPuffBCIMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class sipnPuffBCIMsg : VSMessage
    {

        public double SipnPuffPecent { get; set;}
        public double SipnPuffIntegralPercent { get; set;}

       
        public sipnPuffBCIMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            SipnPuffPecent = 0;
            SipnPuffIntegralPercent = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeDouble(SipnPuffPecent);
            bw.writeDouble(SipnPuffIntegralPercent);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            SipnPuffPecent = br.readDouble();
            SipnPuffIntegralPercent = br.readDouble();

            return br.Idx;
        }
    }
}
