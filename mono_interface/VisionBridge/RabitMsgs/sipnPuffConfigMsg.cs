
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file sipnPuffConfigMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class sipnPuffConfigMsg : VSMessage
    {

        public bool EnableSipnPuffIntegration { get; set;}
        public double SipnPuffBlowGain { get; set;}
        public double SipnPuffSuckGain { get; set;}
        public double SipnPuffDeadBandPercent { get; set;}

       
        public sipnPuffConfigMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            EnableSipnPuffIntegration = false;
            SipnPuffBlowGain = 0;
            SipnPuffSuckGain = 0;
            SipnPuffDeadBandPercent = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeBool(EnableSipnPuffIntegration);
            bw.writeDouble(SipnPuffBlowGain);
            bw.writeDouble(SipnPuffSuckGain);
            bw.writeDouble(SipnPuffDeadBandPercent);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            EnableSipnPuffIntegration = br.readBool();
            SipnPuffBlowGain = br.readDouble();
            SipnPuffSuckGain = br.readDouble();
            SipnPuffDeadBandPercent = br.readDouble();

            return br.Idx;
        }
    }
}
