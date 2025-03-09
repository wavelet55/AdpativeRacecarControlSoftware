#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <RabitManager.h>
#include "VehicleInertialStatesMessage.h"
#include "image_plus_metadata_message.h"


using namespace videre;
using namespace Rabit;


#ifndef VIDERE_DEV_VEHICLEINTERTIALSTATESMSGTESTS_H
#define VIDERE_DEV_VEHICLEINTERTIALSTATESMSGTESTS_H


class VehicleIntertialStatesMsgTests : public ::testing::Test
{
protected:

    // You can do set-up work for each test here.
    VehicleIntertialStatesMsgTests()
    {}

    // You can do clean-up work that doesn't throw exceptions here.
    virtual ~VehicleIntertialStatesMsgTests()
    {}

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    // Code here will be called immediately after the constructor (right
    // before each test).
    virtual void SetUp()
    {}

    // Code here will be called immediately after each test (right
    // before the destructor).
    //virtual void TearDown();

};


#endif //VIDERE_DEV_VEHICLEINTERTIALSTATESMSGTESTS_H
