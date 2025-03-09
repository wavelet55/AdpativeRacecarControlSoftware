
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file nexusBCIControlConfigMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class nexusBCIControlConfigMsg : VSMessage
    {

        public bool EnableNexusBCIThrottleControl { get; set;}
        public bool EnableSipnPuffThrottleBrakeControl { get; set;}
        public bool SipnPuffBrakeOnlyControl { get; set;}
        public double BCIThrottleIntegrationGain { get; set;}
        public double BCIThrottleRampDownDelaySeconds { get; set;}
        public double BCIThrottleRampDownIntegrationGain { get; set;}

       
        public nexusBCIControlConfigMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            EnableNexusBCIThrottleControl = false;
            EnableSipnPuffThrottleBrakeControl = false;
            SipnPuffBrakeOnlyControl = false;
            BCIThrottleIntegrationGain = 0;
            BCIThrottleRampDownDelaySeconds = 0;
            BCIThrottleRampDownIntegrationGain = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeBool(EnableNexusBCIThrottleControl);
            bw.writeBool(EnableSipnPuffThrottleBrakeControl);
            bw.writeBool(SipnPuffBrakeOnlyControl);
            bw.writeDouble(BCIThrottleIntegrationGain);
            bw.writeDouble(BCIThrottleRampDownDelaySeconds);
            bw.writeDouble(BCIThrottleRampDownIntegrationGain);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            EnableNexusBCIThrottleControl = br.readBool();
            EnableSipnPuffThrottleBrakeControl = br.readBool();
            SipnPuffBrakeOnlyControl = br.readBool();
            BCIThrottleIntegrationGain = br.readDouble();
            BCIThrottleRampDownDelaySeconds = br.readDouble();
            BCIThrottleRampDownIntegrationGain = br.readDouble();

            return br.Idx;
        }
    }
}
