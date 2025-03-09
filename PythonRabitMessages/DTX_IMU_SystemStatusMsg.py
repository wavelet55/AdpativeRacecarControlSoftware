
# ****************************************************************
# *******************************************************************

# Auto-generated from file DTX_IMU_SystemStatusMsg.msg, do not modify directly 


import struct

class VSMessage(object):
    """ This is a small enough base class that we are including it in all messages

    Including this in all messages means each python message doesn't have any dependencies
    """

    hfmt = '>iiid'

    def __init__(self):
        self.MessageType = 0
        self.MessageSource = 0
        self.MessageDestination = 0
        self.TimeStampSec = 0
        self.hs = struct.Struct(self.hfmt)

    def getMsgHeaderSizeInBytes(self):
        return 3*4 + 8   # 3 * sizeof(Int32) + sizeof(double);

    def SerializeHeader(self):
        data = (self.MessageType, self.MessageSource, self.MessageDestination, self.TimeStampSec)
        return self.hs.pack(*data)
        
    def Serialize(self, serial_type = "DtiByteArray"):
        return

    def DeserializeHeader(self, byte_array):
        vals = self.hs.unpack(byte_array)
        self.MessageType = vals[0]
        self.MessageSource = vals[1]
        self.MessageDestination = vals[2]
        self.TimeStampSec = vals[3]

    def Deserialize(self, byte_array, serial_type = "DtiByteArray"):
        return



class DTX_IMU_SystemStatusMsg(VSMessage):
    """ Auto generated message for DTX_IMU_SystemStatusMsg
    """

    def __init__(self):
        super().__init__()

        self.fmt = '>'
        self.fmt += 'Q '
        self.fmt += 'B '
        self.fmt += 'B '
        self.fmt += 'B '
        self.fmt += 'B '
        self.fmt += '? '
        self.fmt += '? '
        self.fmt += '? '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '

        self.s = struct.Struct(self.fmt)

        self._msg = {}
        self.Clear()

    def Clear(self):

        self._msg['timestamp'] = {"type": "uint64", "value": 0 }
        self._msg['gf360_cmd_state'] = {"type": "uint8", "value": 0 }
        self._msg['switch_states'] = {"type": "uint8", "value": 0 }
        self._msg['exercise_init_state'] = {"type": "uint8", "value": 0 }
        self._msg['exercise_user_situate_state'] = {"type": "uint8", "value": 0 }
        self._msg['main_air_solenoid_state'] = {"type": "bool", "value": False }
        self._msg['main_cylender_port_a_state'] = {"type": "bool", "value": False }
        self._msg['main_cylender_port_b_state'] = {"type": "bool", "value": False }
        self._msg['main_resistance_regulator_pressure'] = {"type": "float32", "value": 0 }
        self._msg['counter_ballance_regulator_pressure'] = {"type": "float32", "value": 0 }
        self._msg['spare_analog_input'] = {"type": "float32", "value": 0 }
        self._msg['left_motor_position'] = {"type": "float32", "value": 0 }
        self._msg['right_motor_position'] = {"type": "float32", "value": 0 }
        self._msg['harness_updwn_position'] = {"type": "float32", "value": 0 }
        self._msg['harness_left_right_position'] = {"type": "float32", "value": 0 }
        self._msg['harness_rotation_position'] = {"type": "float32", "value": 0 }


    @property
    def timestamp(self):
        return self._msg["timestamp"]["value"]

    @timestamp.setter
    def timestamp(self, val):
        self._msg["timestamp"]["value"] = val

    @property
    def gf360_cmd_state(self):
        return self._msg["gf360_cmd_state"]["value"]

    @gf360_cmd_state.setter
    def gf360_cmd_state(self, val):
        self._msg["gf360_cmd_state"]["value"] = val

    @property
    def switch_states(self):
        return self._msg["switch_states"]["value"]

    @switch_states.setter
    def switch_states(self, val):
        self._msg["switch_states"]["value"] = val

    @property
    def exercise_init_state(self):
        return self._msg["exercise_init_state"]["value"]

    @exercise_init_state.setter
    def exercise_init_state(self, val):
        self._msg["exercise_init_state"]["value"] = val

    @property
    def exercise_user_situate_state(self):
        return self._msg["exercise_user_situate_state"]["value"]

    @exercise_user_situate_state.setter
    def exercise_user_situate_state(self, val):
        self._msg["exercise_user_situate_state"]["value"] = val

    @property
    def main_air_solenoid_state(self):
        return self._msg["main_air_solenoid_state"]["value"]

    @main_air_solenoid_state.setter
    def main_air_solenoid_state(self, val):
        self._msg["main_air_solenoid_state"]["value"] = val

    @property
    def main_cylender_port_a_state(self):
        return self._msg["main_cylender_port_a_state"]["value"]

    @main_cylender_port_a_state.setter
    def main_cylender_port_a_state(self, val):
        self._msg["main_cylender_port_a_state"]["value"] = val

    @property
    def main_cylender_port_b_state(self):
        return self._msg["main_cylender_port_b_state"]["value"]

    @main_cylender_port_b_state.setter
    def main_cylender_port_b_state(self, val):
        self._msg["main_cylender_port_b_state"]["value"] = val

    @property
    def main_resistance_regulator_pressure(self):
        return self._msg["main_resistance_regulator_pressure"]["value"]

    @main_resistance_regulator_pressure.setter
    def main_resistance_regulator_pressure(self, val):
        self._msg["main_resistance_regulator_pressure"]["value"] = val

    @property
    def counter_ballance_regulator_pressure(self):
        return self._msg["counter_ballance_regulator_pressure"]["value"]

    @counter_ballance_regulator_pressure.setter
    def counter_ballance_regulator_pressure(self, val):
        self._msg["counter_ballance_regulator_pressure"]["value"] = val

    @property
    def spare_analog_input(self):
        return self._msg["spare_analog_input"]["value"]

    @spare_analog_input.setter
    def spare_analog_input(self, val):
        self._msg["spare_analog_input"]["value"] = val

    @property
    def left_motor_position(self):
        return self._msg["left_motor_position"]["value"]

    @left_motor_position.setter
    def left_motor_position(self, val):
        self._msg["left_motor_position"]["value"] = val

    @property
    def right_motor_position(self):
        return self._msg["right_motor_position"]["value"]

    @right_motor_position.setter
    def right_motor_position(self, val):
        self._msg["right_motor_position"]["value"] = val

    @property
    def harness_updwn_position(self):
        return self._msg["harness_updwn_position"]["value"]

    @harness_updwn_position.setter
    def harness_updwn_position(self, val):
        self._msg["harness_updwn_position"]["value"] = val

    @property
    def harness_left_right_position(self):
        return self._msg["harness_left_right_position"]["value"]

    @harness_left_right_position.setter
    def harness_left_right_position(self, val):
        self._msg["harness_left_right_position"]["value"] = val

    @property
    def harness_rotation_position(self):
        return self._msg["harness_rotation_position"]["value"]

    @harness_rotation_position.setter
    def harness_rotation_position(self, val):
        self._msg["harness_rotation_position"]["value"] = val



    def __str__(self):
        msg = [self.MessageType, self.MessageSource, self.MessageDestination, self.TimeStampSec, self._msg]
        return str(msg)

    def ToString(self):
        return str(self)

    def Serialize(self, type = "DtiByteArray"):

        byte_header = self.SerializeHeader()

        values = []
        for key in self._msg:
            if "length" in self._msg[key]:
                for val in self._msg[key]["value"]:
                    values.append(val) 
            else:
                values.append(self._msg[key]["value"])

        byte_body = self.s.pack(*values)

        barray = bytearray(byte_header) + bytearray(byte_body)

        return bytes(barray)

    def Deserialize(self, byte_array, type = "DtiByteArray"):
        
        length = self.getMsgHeaderSizeInBytes()
        header = byte_array[0:length]
        body = byte_array[length:]
        self.DeserializeHeader(header)
        vals = self.s.unpack(body)

        self._msg['timestamp']["value"] = vals[0]
        self._msg['gf360_cmd_state']["value"] = vals[1]
        self._msg['switch_states']["value"] = vals[2]
        self._msg['exercise_init_state']["value"] = vals[3]
        self._msg['exercise_user_situate_state']["value"] = vals[4]
        self._msg['main_air_solenoid_state']["value"] = vals[5]
        self._msg['main_cylender_port_a_state']["value"] = vals[6]
        self._msg['main_cylender_port_b_state']["value"] = vals[7]
        self._msg['main_resistance_regulator_pressure']["value"] = vals[8]
        self._msg['counter_ballance_regulator_pressure']["value"] = vals[9]
        self._msg['spare_analog_input']["value"] = vals[10]
        self._msg['left_motor_position']["value"] = vals[11]
        self._msg['right_motor_position']["value"] = vals[12]
        self._msg['harness_updwn_position']["value"] = vals[13]
        self._msg['harness_left_right_position']["value"] = vals[14]
        self._msg['harness_rotation_position']["value"] = vals[15]

    