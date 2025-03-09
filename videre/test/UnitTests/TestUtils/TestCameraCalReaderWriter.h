//
// Created by wavelet on 3/29/17.
//

#ifndef VIDERE_DEV_TESTCAMERACALREADERWRITER_H
#define VIDERE_DEV_TESTCAMERACALREADERWRITER_H

#include "gtest/gtest.h"
#include "FileUtils.h"

class TestCameraCalReaderWriter   : public ::testing::Test
{
protected:

    // You can do set-up work for each test here.
    TestCameraCalReaderWriter()
    {}

    // You can do clean-up work that doesn't throw exceptions here.
    virtual ~TestCameraCalReaderWriter()
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


#endif //VIDERE_DEV_TESTCAMERACALREADERWRITER_H
