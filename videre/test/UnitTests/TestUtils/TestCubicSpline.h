//
// Created by nvidia on 7/17/18.
//

#ifndef VIDERE_DEV_TESTCUBICSPLINE_H
#define VIDERE_DEV_TESTCUBICSPLINE_H

#include "gtest/gtest.h"
#include "CubicSplineProcessor.h"


class TestCubicSpline  : public ::testing::Test
{
protected:

    // You can do set-up work for each test here.
    TestCubicSpline()
    {}

    // You can do clean-up work that doesn't throw exceptions here.
    virtual ~TestCubicSpline()
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


#endif //VIDERE_DEV_TESTCUBICSPLINE_H
