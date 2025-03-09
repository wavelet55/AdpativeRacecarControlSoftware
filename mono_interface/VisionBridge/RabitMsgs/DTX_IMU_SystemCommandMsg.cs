
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file DTX_IMU_SystemCommandMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class DTX_IMU_SystemCommandMsg : VSMessage
    {

        public UInt64 timestamp { get; set;}
        public UInt8 gf360_cmd_state { get; set;}
        public bool main_air_solenoid_enable { get; set;}
        public bool main_cylinder_port_a_enable { get; set;}
        public bool main_cylinder_port_b_enable { get; set;}
        public float main_resistance_regultor_ctrl { get; set;}
        public float counter_ballance_regultor_ctrl { get; set;}
        public float left_motor_position { get; set;}
        public float right_motor_position { get; set;}
        public float harness_updwn_motor_position { get; set;}

       
        public DTX_IMU_SystemCommandMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            timestamp = 0;
            gf360_cmd_state = 0;
            main_air_solenoid_enable = false;
            main_cylinder_port_a_enable = false;
            main_cylinder_port_b_enable = false;
            main_resistance_regultor_ctrl = 0;
            counter_ballance_regultor_ctrl = 0;
            left_motor_position = 0;
            right_motor_position = 0;
            harness_updwn_motor_position = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeUInt64(timestamp);
            bw.writeUInt8(gf360_cmd_state);
            bw.writeBool(main_air_solenoid_enable);
            bw.writeBool(main_cylinder_port_a_enable);
            bw.writeBool(main_cylinder_port_b_enable);
            bw.writeFloat(main_resistance_regultor_ctrl);
            bw.writeFloat(counter_ballance_regultor_ctrl);
            bw.writeFloat(left_motor_position);
            bw.writeFloat(right_motor_position);
            bw.writeFloat(harness_updwn_motor_position);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            timestamp = br.readUInt64();
            gf360_cmd_state = br.readUInt8();
            main_air_solenoid_enable = br.readBool();
            main_cylinder_port_a_enable = br.readBool();
            main_cylinder_port_b_enable = br.readBool();
            main_resistance_regultor_ctrl = br.readFloat();
            counter_ballance_regultor_ctrl = br.readFloat();
            left_motor_position = br.readFloat();
            right_motor_position = br.readFloat();
            harness_updwn_motor_position = br.readFloat();

            return br.Idx;
        }
    }
}
