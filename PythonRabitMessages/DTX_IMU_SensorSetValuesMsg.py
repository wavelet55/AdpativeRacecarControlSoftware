
# ****************************************************************
# *******************************************************************

# Auto-generated from file DTX_IMU_SensorSetValuesMsg.msg, do not modify directly 


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



class DTX_IMU_SensorSetValuesMsg(VSMessage):
    """ Auto generated message for DTX_IMU_SensorSetValuesMsg
    """

    def __init__(self):
        super().__init__()

        self.fmt = '>'
        self.fmt += 'Q '
        self.fmt += 'B '
        self.fmt += 'B '
        self.fmt += 'B '
        self.fmt += 'B '
        self.fmt += 'B '
        self.fmt += 'B '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '
        self.fmt += 'f '
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
        self._msg['sensor_set_location'] = {"type": "uint8", "value": 0 }
        self._msg['sensor_set_cmd'] = {"type": "uint8", "value": 0 }
        self._msg['accel_cmd_flags'] = {"type": "uint8", "value": 0 }
        self._msg['gyro_cmd_flags'] = {"type": "uint8", "value": 0 }
        self._msg['mag_cmd_flags'] = {"type": "uint8", "value": 0 }
        self._msg['airpres_cmd_flags'] = {"type": "uint8", "value": 0 }
        self._msg['accel_x'] = {"type": "float32", "value": 0 }
        self._msg['accel_y'] = {"type": "float32", "value": 0 }
        self._msg['accel_z'] = {"type": "float32", "value": 0 }
        self._msg['pos_x'] = {"type": "float32", "value": 0 }
        self._msg['pos_y'] = {"type": "float32", "value": 0 }
        self._msg['pos_z'] = {"type": "float32", "value": 0 }
        self._msg['vel_x'] = {"type": "float32", "value": 0 }
        self._msg['vel_y'] = {"type": "float32", "value": 0 }
        self._msg['vel_z'] = {"type": "float32", "value": 0 }
        self._msg['gyro_s'] = {"type": "float32", "value": 0 }
        self._msg['gyro_x'] = {"type": "float32", "value": 0 }
        self._msg['gyro_y'] = {"type": "float32", "value": 0 }
        self._msg['gyro_z'] = {"type": "float32", "value": 0 }
        self._msg['mag_x'] = {"type": "float32", "value": 0 }
        self._msg['mag_y'] = {"type": "float32", "value": 0 }
        self._msg['mag_z'] = {"type": "float32", "value": 0 }
        self._msg['air_pressure'] = {"type": "float32", "value": 0 }
        self._msg['temperature'] = {"type": "float32", "value": 0 }


    @property
    def timestamp(self):
        return self._msg["timestamp"]["value"]

    @timestamp.setter
    def timestamp(self, val):
        self._msg["timestamp"]["value"] = val

    @property
    def sensor_set_location(self):
        return self._msg["sensor_set_location"]["value"]

    @sensor_set_location.setter
    def sensor_set_location(self, val):
        self._msg["sensor_set_location"]["value"] = val

    @property
    def sensor_set_cmd(self):
        return self._msg["sensor_set_cmd"]["value"]

    @sensor_set_cmd.setter
    def sensor_set_cmd(self, val):
        self._msg["sensor_set_cmd"]["value"] = val

    @property
    def accel_cmd_flags(self):
        return self._msg["accel_cmd_flags"]["value"]

    @accel_cmd_flags.setter
    def accel_cmd_flags(self, val):
        self._msg["accel_cmd_flags"]["value"] = val

    @property
    def gyro_cmd_flags(self):
        return self._msg["gyro_cmd_flags"]["value"]

    @gyro_cmd_flags.setter
    def gyro_cmd_flags(self, val):
        self._msg["gyro_cmd_flags"]["value"] = val

    @property
    def mag_cmd_flags(self):
        return self._msg["mag_cmd_flags"]["value"]

    @mag_cmd_flags.setter
    def mag_cmd_flags(self, val):
        self._msg["mag_cmd_flags"]["value"] = val

    @property
    def airpres_cmd_flags(self):
        return self._msg["airpres_cmd_flags"]["value"]

    @airpres_cmd_flags.setter
    def airpres_cmd_flags(self, val):
        self._msg["airpres_cmd_flags"]["value"] = val

    @property
    def accel_x(self):
        return self._msg["accel_x"]["value"]

    @accel_x.setter
    def accel_x(self, val):
        self._msg["accel_x"]["value"] = val

    @property
    def accel_y(self):
        return self._msg["accel_y"]["value"]

    @accel_y.setter
    def accel_y(self, val):
        self._msg["accel_y"]["value"] = val

    @property
    def accel_z(self):
        return self._msg["accel_z"]["value"]

    @accel_z.setter
    def accel_z(self, val):
        self._msg["accel_z"]["value"] = val

    @property
    def pos_x(self):
        return self._msg["pos_x"]["value"]

    @pos_x.setter
    def pos_x(self, val):
        self._msg["pos_x"]["value"] = val

    @property
    def pos_y(self):
        return self._msg["pos_y"]["value"]

    @pos_y.setter
    def pos_y(self, val):
        self._msg["pos_y"]["value"] = val

    @property
    def pos_z(self):
        return self._msg["pos_z"]["value"]

    @pos_z.setter
    def pos_z(self, val):
        self._msg["pos_z"]["value"] = val

    @property
    def vel_x(self):
        return self._msg["vel_x"]["value"]

    @vel_x.setter
    def vel_x(self, val):
        self._msg["vel_x"]["value"] = val

    @property
    def vel_y(self):
        return self._msg["vel_y"]["value"]

    @vel_y.setter
    def vel_y(self, val):
        self._msg["vel_y"]["value"] = val

    @property
    def vel_z(self):
        return self._msg["vel_z"]["value"]

    @vel_z.setter
    def vel_z(self, val):
        self._msg["vel_z"]["value"] = val

    @property
    def gyro_s(self):
        return self._msg["gyro_s"]["value"]

    @gyro_s.setter
    def gyro_s(self, val):
        self._msg["gyro_s"]["value"] = val

    @property
    def gyro_x(self):
        return self._msg["gyro_x"]["value"]

    @gyro_x.setter
    def gyro_x(self, val):
        self._msg["gyro_x"]["value"] = val

    @property
    def gyro_y(self):
        return self._msg["gyro_y"]["value"]

    @gyro_y.setter
    def gyro_y(self, val):
        self._msg["gyro_y"]["value"] = val

    @property
    def gyro_z(self):
        return self._msg["gyro_z"]["value"]

    @gyro_z.setter
    def gyro_z(self, val):
        self._msg["gyro_z"]["value"] = val

    @property
    def mag_x(self):
        return self._msg["mag_x"]["value"]

    @mag_x.setter
    def mag_x(self, val):
        self._msg["mag_x"]["value"] = val

    @property
    def mag_y(self):
        return self._msg["mag_y"]["value"]

    @mag_y.setter
    def mag_y(self, val):
        self._msg["mag_y"]["value"] = val

    @property
    def mag_z(self):
        return self._msg["mag_z"]["value"]

    @mag_z.setter
    def mag_z(self, val):
        self._msg["mag_z"]["value"] = val

    @property
    def air_pressure(self):
        return self._msg["air_pressure"]["value"]

    @air_pressure.setter
    def air_pressure(self, val):
        self._msg["air_pressure"]["value"] = val

    @property
    def temperature(self):
        return self._msg["temperature"]["value"]

    @temperature.setter
    def temperature(self, val):
        self._msg["temperature"]["value"] = val



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
        self._msg['sensor_set_location']["value"] = vals[1]
        self._msg['sensor_set_cmd']["value"] = vals[2]
        self._msg['accel_cmd_flags']["value"] = vals[3]
        self._msg['gyro_cmd_flags']["value"] = vals[4]
        self._msg['mag_cmd_flags']["value"] = vals[5]
        self._msg['airpres_cmd_flags']["value"] = vals[6]
        self._msg['accel_x']["value"] = vals[7]
        self._msg['accel_y']["value"] = vals[8]
        self._msg['accel_z']["value"] = vals[9]
        self._msg['pos_x']["value"] = vals[10]
        self._msg['pos_y']["value"] = vals[11]
        self._msg['pos_z']["value"] = vals[12]
        self._msg['vel_x']["value"] = vals[13]
        self._msg['vel_y']["value"] = vals[14]
        self._msg['vel_z']["value"] = vals[15]
        self._msg['gyro_s']["value"] = vals[16]
        self._msg['gyro_x']["value"] = vals[17]
        self._msg['gyro_y']["value"] = vals[18]
        self._msg['gyro_z']["value"] = vals[19]
        self._msg['mag_x']["value"] = vals[20]
        self._msg['mag_y']["value"] = vals[21]
        self._msg['mag_z']["value"] = vals[22]
        self._msg['air_pressure']["value"] = vals[23]
        self._msg['temperature']["value"] = vals[24]

    