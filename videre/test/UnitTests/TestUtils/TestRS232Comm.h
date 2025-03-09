//
// Created by nvidia on 1/15/18.
//

#ifndef VIDERE_DEV_TESTRS232COMM_H
#define VIDERE_DEV_TESTRS232COMM_H

#include "gtest/gtest.h"
#include "SerialCommMessage.h"
#include "RS232Comm.h"

class TestRS232Comm  : public ::testing::Test
{

protected:
    bool testDone = false;

    // You can do set-up work for each test here.
    TestRS232Comm()
    {}

    // You can do clean-up work that doesn't throw exceptions here.
    virtual ~TestRS232Comm()
    {
        rs232Comm.shutdown();
    }

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    // Code here will be called immediately after the constructor (right
    // before each test).
    virtual void SetUp()
    {}

    // Code here will be called immediately after each test (right
    // before the destructor).
    //virtual void TearDown();

public:
    dtiRS232Comm::RS232Comm rs232Comm;

    //void rxMsgHandler(dtiUtils::SerialCommMessage_t &msg);

};


#endif //VIDERE_DEV_TESTRS232COMM_H
