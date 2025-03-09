#DTX IMU Sensor Command
#Used with the ICM20648 and SensorSetProcessing classes.
#There are three identical message topics of exactly the same structure
#for communicating with the Sensor Sets.
uint64  timestamp	# time since system start (microseconds)
uint8   sensor_set_location         #Ref: dtx::IMU_DeviceLocation_e
uint8   sensor_set_cmd              #Type of processing for the sensor set
uint8   accel_cmd_flags             #Specific control of the Accelerometer Processing
uint8   gyro_cmd_flags             #Specific control of the Gyro Processing
uint8   mag_cmd_flags              #Specific control of the Magetometer Processing
uint8   airpres_cmd_flags          #Specific control of the Air Pressure Processing
