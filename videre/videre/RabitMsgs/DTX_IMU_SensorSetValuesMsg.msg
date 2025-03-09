#DTX IMU Sensor Values
#Used with the ICM20648 and SensorSetProcessing classes.
#There are three identical message topics of exactly the same structure
#for communicating with the Sensor Sets.  The type of data
#in the different values may depend upon the cmd / flags
uint64  timestamp	# time since system start (microseconds)
uint8   sensor_set_location         #Ref: dtx::IMU_DeviceLocation_e
uint8   sensor_set_cmd              #Type of processing for the sensor set
uint8   accel_cmd_flags             #Specific control of the Accelerometer Processing
uint8   gyro_cmd_flags             #Specific control of the Gyro Processing
uint8   mag_cmd_flags              #Specific control of the Magetometer Processing
uint8   airpres_cmd_flags          #Specific control of the Air Pressure Processing
float32 accel_x
float32 accel_y
float32 accel_z
#Position and velocity may be used to output various values such as sensor statistics
#depending on thesensor_set_cmd and individual sensor commands.
float32 pos_x
float32 pos_y
float32 pos_z
float32 vel_x
float32 vel_y
float32 vel_z
#The gyro output may be actual gyro values or a quaternion.
float32 gyro_s
float32 gyro_x
float32 gyro_y
float32 gyro_z
float32 mag_x
float32 mag_y
float32 mag_z
float32 air_pressure
float32 temperature
