
# ****************************************************************
# *******************************************************************

# Auto-generated from file DTX_IMU_SensorSetCommandMsg.msg, do not modify directly 


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



class DTX_IMU_SensorSetCommandMsg(VSMessage):
    """ Auto generated message for DTX_IMU_SensorSetCommandMsg
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

    