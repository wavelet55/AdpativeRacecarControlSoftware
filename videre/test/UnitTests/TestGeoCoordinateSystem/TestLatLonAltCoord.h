//
// Created by wavelet on 8/17/16.
//
#ifndef TEST_LAT_LON_ALT_COORD_HEADER
#define TEST_LAT_LON_ALT_COORD_HEADER

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <RabitManager.h>
#include "TestLatLonAltCoord.h"

class TestLatLonAltCoord : public ::testing::Test
{
protected:

    // You can do set-up work for each test here.
    TestLatLonAltCoord()
    {}

    // You can do clean-up work that doesn't throw exceptions here.
    virtual ~TestLatLonAltCoord()
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

#endif  //TEST_LAT_LON_ALT_COORD_HEADER