"""

Randy Direen
8/2/2018

"""
# -------------------------------------------------------------------Built-ins
# -------------------------------------------------------------------3rd Party
import numpy as np
import quaternion as qt
# ----------------------------------------------------------------------Custom


class VisionData:
    def __init__(self):
        """
        Put the raw data from the processed camera data into an object

        @param raw_data:
        """
        self.time_sec = 0.0
        self.cam_sec = 0.0
        self.valid = False
        self.q = qt.quaternion(1, 0, 0, 0)
        self.r = np.array([0, 0, 0], np.float64)
        self.image_number = 0

    def from_gpb(self, gpb_message):

        # rvec = np.array([gpb_message.HeadOrientationRVec_X,
        #                  gpb_message.HeadOrientationRVec_Y,
        #                  gpb_message.HeadOrientationRVec_Z], np.float64)
        # self.q = qt.from_rotation_vector(rvec)

        self.q.w = gpb_message.HeadOrientationQuaternion_W
        self.q.x = gpb_message.HeadOrientationQuaternion_X
        self.q.y = gpb_message.HeadOrientationQuaternion_Y
        self.q.z = gpb_message.HeadOrientationQuaternion_Z

        self.r[0] = gpb_message.HeadTranslationVec_X
        self.r[1] = gpb_message.HeadTranslationVec_Y
        self.r[2] = gpb_message.HeadTranslationVec_Z

        self.valid = gpb_message.IsDataValid

        self.image_number = gpb_message.ImageNumber

        self.time_sec = gpb_message.VidereTimeStampSec

        self.cam_sec = gpb_message.ImageCaptureTimeStampSec

        self.cov_norm = gpb_message.CovarianceNorm

    def from_file(self, file_message):
        pass

    def __repr__(self):
        ss = """TimeV[{0:16.4f}] TimeCam[{1:16.4f}] IsValid[{2}] ImgNum[{3}]
qw = {4:4.2f}, qx = {5:4.2f}, qy = {6:4.2f}, qz = {7:4.2},
x = {8:4.2f}, y = {9:4.2f}, z = {10:4.2f}"""

        return ss.format(self.time_sec,
                         self.cam_sec,
                         self.valid,
                         self.image_number,
                         self.q.w, self.q.x, self.q.y, self.q.z,
                         self.r[0], self.r[1], self.r[2])



class IMUData:
    def __init__(self):
        """
        Put the raw IMU data point into an object

        @param raw_data:
        """
        self.time_sec = 0.0
        self.imu_sec = 0.0
        self.accel = np.array([0, 0, 0], np.float64)
        self.gyro = np.array([0, 0, 0], np.float64)

    def from_gpb(self, gpb_message):

        self.accel[0] = gpb_message.AccelMPS2_X
        self.accel[1] = gpb_message.AccelMPS2_Y
        self.accel[2] = gpb_message.AccelMPS2_Z

        self.gyro[0] = gpb_message.GyroRadPerSec_X
        self.gyro[1] = gpb_message.GyroRadPerSec_Y
        self.gyro[2] = gpb_message.GyroRadPerSec_Z

        self.time_sec = gpb_message.VidereTimeStampSec

        self.imu_sec = gpb_message.IMUTimeStampSec

    def __repr__(self):
        ss = """TimeV[{0:16.4f}] TimeIMU[{1:16.4f}] 
ax = {2:4.2f}, ay = {3:4.2f}, az = {4:4.2f}
wx = {5:4.2f}, wy = {6:4.2f}, wz = {7:4.2f}"""

        return ss.format(self.time_sec, self.imu_sec,
                         self.accel[0], self.accel[1], self.accel[2],
                         self.gyro[0], self.gyro[1], self.gyro[2])
