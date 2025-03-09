
# ****************************************************************
# *******************************************************************

# Auto-generated from file nexusBCIControlConfigMsg.msg, do not modify directly 


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



class nexusBCIControlConfigMsg(VSMessage):
    """ Auto generated message for nexusBCIControlConfigMsg
    """

    def __init__(self):
        super().__init__()

        self.fmt = '>'
        self.fmt += '? '
        self.fmt += '? '
        self.fmt += '? '
        self.fmt += 'd '
        self.fmt += 'd '
        self.fmt += 'd '

        self.s = struct.Struct(self.fmt)

        self._msg = {}
        self.Clear()

    def Clear(self):

        self._msg['EnableNexusBCIThrottleControl'] = {"type": "bool", "value": False }
        self._msg['EnableSipnPuffThrottleBrakeControl'] = {"type": "bool", "value": False }
        self._msg['SipnPuffBrakeOnlyControl'] = {"type": "bool", "value": False }
        self._msg['BCIThrottleIntegrationGain'] = {"type": "float64", "value": 0 }
        self._msg['BCIThrottleRampDownDelaySeconds'] = {"type": "float64", "value": 0 }
        self._msg['BCIThrottleRampDownIntegrationGain'] = {"type": "float64", "value": 0 }


    @property
    def EnableNexusBCIThrottleControl(self):
        return self._msg["EnableNexusBCIThrottleControl"]["value"]

    @EnableNexusBCIThrottleControl.setter
    def EnableNexusBCIThrottleControl(self, val):
        self._msg["EnableNexusBCIThrottleControl"]["value"] = val

    @property
    def EnableSipnPuffThrottleBrakeControl(self):
        return self._msg["EnableSipnPuffThrottleBrakeControl"]["value"]

    @EnableSipnPuffThrottleBrakeControl.setter
    def EnableSipnPuffThrottleBrakeControl(self, val):
        self._msg["EnableSipnPuffThrottleBrakeControl"]["value"] = val

    @property
    def SipnPuffBrakeOnlyControl(self):
        return self._msg["SipnPuffBrakeOnlyControl"]["value"]

    @SipnPuffBrakeOnlyControl.setter
    def SipnPuffBrakeOnlyControl(self, val):
        self._msg["SipnPuffBrakeOnlyControl"]["value"] = val

    @property
    def BCIThrottleIntegrationGain(self):
        return self._msg["BCIThrottleIntegrationGain"]["value"]

    @BCIThrottleIntegrationGain.setter
    def BCIThrottleIntegrationGain(self, val):
        self._msg["BCIThrottleIntegrationGain"]["value"] = val

    @property
    def BCIThrottleRampDownDelaySeconds(self):
        return self._msg["BCIThrottleRampDownDelaySeconds"]["value"]

    @BCIThrottleRampDownDelaySeconds.setter
    def BCIThrottleRampDownDelaySeconds(self, val):
        self._msg["BCIThrottleRampDownDelaySeconds"]["value"] = val

    @property
    def BCIThrottleRampDownIntegrationGain(self):
        return self._msg["BCIThrottleRampDownIntegrationGain"]["value"]

    @BCIThrottleRampDownIntegrationGain.setter
    def BCIThrottleRampDownIntegrationGain(self, val):
        self._msg["BCIThrottleRampDownIntegrationGain"]["value"] = val



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

        self._msg['EnableNexusBCIThrottleControl']["value"] = vals[0]
        self._msg['EnableSipnPuffThrottleBrakeControl']["value"] = vals[1]
        self._msg['SipnPuffBrakeOnlyControl']["value"] = vals[2]
        self._msg['BCIThrottleIntegrationGain']["value"] = vals[3]
        self._msg['BCIThrottleRampDownDelaySeconds']["value"] = vals[4]
        self._msg['BCIThrottleRampDownIntegrationGain']["value"] = vals[5]

    