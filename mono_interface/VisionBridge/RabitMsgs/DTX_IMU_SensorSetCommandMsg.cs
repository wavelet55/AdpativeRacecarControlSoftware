
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file DTX_IMU_SensorSetCommandMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class DTX_IMU_SensorSetCommandMsg : VSMessage
    {

        public UInt64 timestamp { get; set;}
        public UInt8 sensor_set_location { get; set;}
        public UInt8 sensor_set_cmd { get; set;}
        public UInt8 accel_cmd_flags { get; set;}
        public UInt8 gyro_cmd_flags { get; set;}
        public UInt8 mag_cmd_flags { get; set;}
        public UInt8 airpres_cmd_flags { get; set;}

       
        public DTX_IMU_SensorSetCommandMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            timestamp = 0;
            sensor_set_location = 0;
            sensor_set_cmd = 0;
            accel_cmd_flags = 0;
            gyro_cmd_flags = 0;
            mag_cmd_flags = 0;
            airpres_cmd_flags = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeUInt64(timestamp);
            bw.writeUInt8(sensor_set_location);
            bw.writeUInt8(sensor_set_cmd);
            bw.writeUInt8(accel_cmd_flags);
            bw.writeUInt8(gyro_cmd_flags);
            bw.writeUInt8(mag_cmd_flags);
            bw.writeUInt8(airpres_cmd_flags);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            timestamp = br.readUInt64();
            sensor_set_location = br.readUInt8();
            sensor_set_cmd = br.readUInt8();
            accel_cmd_flags = br.readUInt8();
            gyro_cmd_flags = br.readUInt8();
            mag_cmd_flags = br.readUInt8();
            airpres_cmd_flags = br.readUInt8();

            return br.Idx;
        }
    }
}
