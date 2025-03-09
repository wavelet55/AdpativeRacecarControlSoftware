
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file DTX_IMU_SensorSetValuesMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class DTX_IMU_SensorSetValuesMsg : VSMessage
    {

        public UInt64 timestamp { get; set;}
        public UInt8 sensor_set_location { get; set;}
        public UInt8 sensor_set_cmd { get; set;}
        public UInt8 accel_cmd_flags { get; set;}
        public UInt8 gyro_cmd_flags { get; set;}
        public UInt8 mag_cmd_flags { get; set;}
        public UInt8 airpres_cmd_flags { get; set;}
        public float accel_x { get; set;}
        public float accel_y { get; set;}
        public float accel_z { get; set;}
        public float pos_x { get; set;}
        public float pos_y { get; set;}
        public float pos_z { get; set;}
        public float vel_x { get; set;}
        public float vel_y { get; set;}
        public float vel_z { get; set;}
        public float gyro_s { get; set;}
        public float gyro_x { get; set;}
        public float gyro_y { get; set;}
        public float gyro_z { get; set;}
        public float mag_x { get; set;}
        public float mag_y { get; set;}
        public float mag_z { get; set;}
        public float air_pressure { get; set;}
        public float temperature { get; set;}

       
        public DTX_IMU_SensorSetValuesMsg()
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
            accel_x = 0;
            accel_y = 0;
            accel_z = 0;
            pos_x = 0;
            pos_y = 0;
            pos_z = 0;
            vel_x = 0;
            vel_y = 0;
            vel_z = 0;
            gyro_s = 0;
            gyro_x = 0;
            gyro_y = 0;
            gyro_z = 0;
            mag_x = 0;
            mag_y = 0;
            mag_z = 0;
            air_pressure = 0;
            temperature = 0;

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
            bw.writeFloat(accel_x);
            bw.writeFloat(accel_y);
            bw.writeFloat(accel_z);
            bw.writeFloat(pos_x);
            bw.writeFloat(pos_y);
            bw.writeFloat(pos_z);
            bw.writeFloat(vel_x);
            bw.writeFloat(vel_y);
            bw.writeFloat(vel_z);
            bw.writeFloat(gyro_s);
            bw.writeFloat(gyro_x);
            bw.writeFloat(gyro_y);
            bw.writeFloat(gyro_z);
            bw.writeFloat(mag_x);
            bw.writeFloat(mag_y);
            bw.writeFloat(mag_z);
            bw.writeFloat(air_pressure);
            bw.writeFloat(temperature);

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
            accel_x = br.readFloat();
            accel_y = br.readFloat();
            accel_z = br.readFloat();
            pos_x = br.readFloat();
            pos_y = br.readFloat();
            pos_z = br.readFloat();
            vel_x = br.readFloat();
            vel_y = br.readFloat();
            vel_z = br.readFloat();
            gyro_s = br.readFloat();
            gyro_x = br.readFloat();
            gyro_y = br.readFloat();
            gyro_z = br.readFloat();
            mag_x = br.readFloat();
            mag_y = br.readFloat();
            mag_z = br.readFloat();
            air_pressure = br.readFloat();
            temperature = br.readFloat();

            return br.Idx;
        }
    }
}
