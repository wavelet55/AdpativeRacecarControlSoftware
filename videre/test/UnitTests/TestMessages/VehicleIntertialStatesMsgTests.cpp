//
// Created by wavelet on 8/22/16.
//

#include "VehicleIntertialStatesMsgTests.h"
#include <memory>
#include <string>

using namespace std;



TEST_F(VehicleIntertialStatesMsgTests, TestCopyAndCloneMethod)
{
    VehicleInertialStatesMessage VIS1Msg;
    VehicleInertialStatesMessage VIS2Msg;

    VIS1Msg.LatLonAlt.SetLatitudeDegrees(39.5);
    VIS1Msg.LatLonAlt.SetLongitudeDegrees(-85.75);
    VIS1Msg.LatLonAlt.SetAltitude(2225.0);

    VIS1Msg.XYZCoordinates.x = 100.25;
    VIS1Msg.XYZCoordinates.y = -25.5;
    VIS1Msg.XYZCoordinates.z = 2036.5;

    VIS1Msg.XYZVelocities.x = 22.5;
    VIS1Msg.XYZVelocities.y = -17.25;
    VIS1Msg.XYZVelocities.z = 3.5;

    VIS1Msg.RollPitchYawRates.IsRate = false;
    VIS1Msg.RollPitchYaw.SetRollDegrees(10.5);
    VIS1Msg.RollPitchYaw.SetPitchDegrees(-3.25);
    VIS1Msg.RollPitchYaw.SetRollDegrees(10.5);

    VIS1Msg.RollPitchYawRates.IsRate = true;
    VIS1Msg.RollPitchYawRates.SetRollDegrees(3.75);
    VIS1Msg.RollPitchYawRates.SetPitchDegrees(-8.5);
    VIS1Msg.RollPitchYawRates.SetRollDegrees(2.75);

    VIS1Msg.GpsTimeStampSec = 123456.5;
    VIS1Msg.SetTimeNow();

    VIS2Msg.CopyMessage(&VIS1Msg);

    EXPECT_EQ(VIS1Msg.GetTimeStamp(), VIS2Msg.GetTimeStamp());
    double ts1 = VIS1Msg.TimeStampSeconds();
    double ts2 = VIS2Msg.TimeStampSeconds();
    EXPECT_EQ(ts1, ts2);

    EXPECT_EQ(VIS1Msg.GpsTimeStampSec, VIS2Msg.GpsTimeStampSec);

    EXPECT_EQ(VIS1Msg.LatLonAlt.Altitude(), VIS2Msg.LatLonAlt.Altitude());

    double dist = VIS1Msg.XYZCoordinates.Distance(VIS2Msg.XYZCoordinates);
    int eq = dist < 0.001 ? 0 : 1;
    EXPECT_EQ(eq, 0);

    dist = VIS1Msg.XYZVelocities.Distance(VIS2Msg.XYZVelocities);
    eq = dist < 0.001 ? 0 : 1;
    EXPECT_EQ(eq, 0);

    dist = VIS1Msg.RollPitchYaw.EuclideanDistance(VIS2Msg.RollPitchYaw);
    eq = dist < 0.001 ? 0 : 1;
    EXPECT_EQ(eq, 0);

    dist = VIS1Msg.RollPitchYawRates.EuclideanDistance(VIS2Msg.RollPitchYawRates);
    eq = dist < 0.001 ? 0 : 1;
    EXPECT_EQ(eq, 0);

    std::unique_ptr<Rabit::RabitMessage> rmc = VIS1Msg.Clone();

    VehicleInertialStatesMessage *vmc = static_cast<VehicleInertialStatesMessage *>(rmc.release());

    EXPECT_EQ(VIS1Msg.GetTimeStamp(), VIS2Msg.GetTimeStamp());
    ts1 = VIS1Msg.TimeStampSeconds();
    ts2 = VIS2Msg.TimeStampSeconds();
    EXPECT_EQ(ts1, ts2);

    EXPECT_EQ(VIS1Msg.GpsTimeStampSec, vmc->GpsTimeStampSec);

    EXPECT_EQ(VIS1Msg.LatLonAlt.Altitude(), vmc->LatLonAlt.Altitude());

    dist = VIS1Msg.XYZCoordinates.Distance(vmc->XYZCoordinates);
    eq = dist < 0.001 ? 0 : 1;
    EXPECT_EQ(eq, 0);

    dist = VIS1Msg.XYZVelocities.Distance(vmc->XYZVelocities);
    eq = dist < 0.001 ? 0 : 1;
    EXPECT_EQ(eq, 0);

    dist = VIS1Msg.RollPitchYaw.EuclideanDistance(vmc->RollPitchYaw);
    eq = dist < 0.001 ? 0 : 1;
    EXPECT_EQ(eq, 0);

    dist = VIS1Msg.RollPitchYawRates.EuclideanDistance(vmc->RollPitchYawRates);
    eq = dist < 0.001 ? 0 : 1;
    EXPECT_EQ(eq, 0);


}