
/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file DTX_IMU_SystemStatusMsg.msg, do not modify directly */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Int8 = System.SByte;
using UInt8 = System.Byte;


namespace VisionBridge.Messages
{
    public class DTX_IMU_SystemStatusMsg : VSMessage
    {

        public UInt64 timestamp { get; set;}
        public UInt8 gf360_cmd_state { get; set;}
        public UInt8 switch_states { get; set;}
        public UInt8 exercise_init_state { get; set;}
        public UInt8 exercise_user_situate_state { get; set;}
        public bool main_air_solenoid_state { get; set;}
        public bool main_cylender_port_a_state { get; set;}
        public bool main_cylender_port_b_state { get; set;}
        public float main_resistance_regulator_pressure { get; set;}
        public float counter_ballance_regulator_pressure { get; set;}
        public float spare_analog_input { get; set;}
        public float left_motor_position { get; set;}
        public float right_motor_position { get; set;}
        public float harness_updwn_position { get; set;}
        public float harness_left_right_position { get; set;}
        public float harness_rotation_position { get; set;}

       
        public DTX_IMU_SystemStatusMsg()
        {
            Clear();
        }

        public override void Clear()
        {
            base.Clear();
            timestamp = 0;
            gf360_cmd_state = 0;
            switch_states = 0;
            exercise_init_state = 0;
            exercise_user_situate_state = 0;
            main_air_solenoid_state = false;
            main_cylender_port_a_state = false;
            main_cylender_port_b_state = false;
            main_resistance_regulator_pressure = 0;
            counter_ballance_regulator_pressure = 0;
            spare_analog_input = 0;
            left_motor_position = 0;
            right_motor_position = 0;
            harness_updwn_position = 0;
            harness_left_right_position = 0;
            harness_rotation_position = 0;

        }


        public override byte[] Serialize(MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteListArrayWriter bw = new ByteListArrayWriter(EndianOrder_e.big);
            SerializeHeader(bw);
            bw.writeUInt64(timestamp);
            bw.writeUInt8(gf360_cmd_state);
            bw.writeUInt8(switch_states);
            bw.writeUInt8(exercise_init_state);
            bw.writeUInt8(exercise_user_situate_state);
            bw.writeBool(main_air_solenoid_state);
            bw.writeBool(main_cylender_port_a_state);
            bw.writeBool(main_cylender_port_b_state);
            bw.writeFloat(main_resistance_regulator_pressure);
            bw.writeFloat(counter_ballance_regulator_pressure);
            bw.writeFloat(spare_analog_input);
            bw.writeFloat(left_motor_position);
            bw.writeFloat(right_motor_position);
            bw.writeFloat(harness_updwn_position);
            bw.writeFloat(harness_left_right_position);
            bw.writeFloat(harness_rotation_position);

            return bw.ByteArray;
        }

        public override int Deserialize(byte[] b, MsgSerializationType_e stype = MsgSerializationType_e.DtiByteArray)
        {
            ByteArrayReader br = new ByteArrayReader(b, EndianOrder_e.big);
            DeserializeHeader(br);
            timestamp = br.readUInt64();
            gf360_cmd_state = br.readUInt8();
            switch_states = br.readUInt8();
            exercise_init_state = br.readUInt8();
            exercise_user_situate_state = br.readUInt8();
            main_air_solenoid_state = br.readBool();
            main_cylender_port_a_state = br.readBool();
            main_cylender_port_b_state = br.readBool();
            main_resistance_regulator_pressure = br.readFloat();
            counter_ballance_regulator_pressure = br.readFloat();
            spare_analog_input = br.readFloat();
            left_motor_position = br.readFloat();
            right_motor_position = br.readFloat();
            harness_updwn_position = br.readFloat();
            harness_left_right_position = br.readFloat();
            harness_rotation_position = br.readFloat();

            return br.Idx;
        }
    }
}
