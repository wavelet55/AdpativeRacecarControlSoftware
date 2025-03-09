//
// Created by nvidia on 1/15/18.
//

#include "TestRS232Comm.h"
#include "RS232Comm.h"
#include <iostream>
#include <thread>


using namespace dtiUtils;
using namespace dtiRS232Comm;
using namespace std;

TEST_F(TestRS232Comm, TestProcessTextReadBuffer)
{
    bool tstFlg;
    int tstInt;
    int msgsize = 0;
    int stMsgSize = 0;
    u_char msgBuf[25];

    RS232Comm rs232CommLcl;
    rs232CommLcl.MessageProcessType = RS232Comm_MessageProcessType::RS232CommMPT_TextCmds;
    rs232CommLcl.setRxMsgQueueSize(10);
    rs232CommLcl.setMaxRxBufferSize(256);
    rs232CommLcl.setupRxBuffersAndQueue();

    string msg1 = "Tst=123\n\r";
    msgsize = rs232CommLcl.testOnlyAddMsgToRxBuf((char*)msg1.c_str(), msg1.size());
    rs232CommLcl.processReadBuffer(msgsize);

    stMsgSize = rs232CommLcl.testOnlyGetMsgFromRxQueue((char*)msgBuf);
    EXPECT_EQ(stMsgSize, 7);

    msg1 = "\nRed=456\n\rCmd=";
    msgsize = rs232CommLcl.testOnlyAddMsgToRxBuf((char*)msg1.c_str(), msg1.size());
    rs232CommLcl.processReadBuffer(msgsize);
    stMsgSize = rs232CommLcl.testOnlyGetMsgFromRxQueue((char*)msgBuf);
    EXPECT_EQ(stMsgSize, 7);
    stMsgSize = rs232CommLcl.testOnlyGetMsgFromRxQueue((char*)msgBuf, false);
    EXPECT_EQ(stMsgSize, 4);

    msg1 = "Help\nFin=Done\r   ";
    msgsize = rs232CommLcl.testOnlyAddMsgToRxBuf((char*)msg1.c_str(), msg1.size());
    rs232CommLcl.processReadBuffer(msgsize);
    stMsgSize = rs232CommLcl.testOnlyGetMsgFromRxQueue((char*)msgBuf);
    EXPECT_EQ(stMsgSize, 8);
    stMsgSize = rs232CommLcl.testOnlyGetMsgFromRxQueue((char*)msgBuf, true);
    EXPECT_EQ(stMsgSize, 8);

}

u_char TestRS232CommMsgBuf[256];
int TestRS232CommMsgSize = 0;

void rxMsgHandler(dtiUtils::SerialCommMessage_t &msg, void * pasrserObj)
{
    TestRS232CommMsgSize = msg.getMsg(TestRS232CommMsgBuf);

    cout << TestRS232CommMsgBuf << endl;

    //add a carrage return line feed to the message
    TestRS232CommMsgBuf[TestRS232CommMsgSize++] = '\r';
    TestRS232CommMsgBuf[TestRS232CommMsgSize++] = '\n';
    TestRS232CommMsgBuf[TestRS232CommMsgSize] = '0';

}


//Test requires a valid comm port setup and
//expects a user to send messages which are read and
//echoed back.  A message that starts with Q will cause the
//system to stop.
TEST_F(TestRS232Comm, TestRS232LoopBack)
{
    bool tstFlg;
    int tstInt;
    int msgsize = 0;
    int stMsgSize = 0;
    u_char msgBuf[25];

    testDone = false;
    RS232Comm rs232Comm;
    rs232Comm.MessageProcessType = RS232Comm_MessageProcessType::RS232CommMPT_TextCmds;
    rs232Comm.setRxMsgQueueSize(10);
    rs232Comm.setMaxRxBufferSize(256);
    //rs232Comm.setupRxBuffersAndQueue();
    rs232Comm.ReceiveMessageHandler = &rxMsgHandler;
    rs232Comm.setBaudRate(9600);
    rs232Comm.CommPort = "/dev/ttyUSB0";

    rs232Comm.start();
    string stMsg = "RS232 Comm Test Starting.\r\n";
    rs232Comm.transmitMessage(stMsg);

    while(!testDone)
    {
        rs232Comm.processReceivedMessages();
        if(TestRS232CommMsgSize > 0)
        {
            rs232Comm.transmitMessage(TestRS232CommMsgBuf, TestRS232CommMsgSize);
            if(TestRS232CommMsgBuf[0] == 'Q')
            {
                testDone = true;
            }
            TestRS232CommMsgSize = 0;
        }
    }

    stMsg = "RS232 Comm Test is Done.\r\n";
    rs232Comm.transmitMessage(stMsg);

    std::this_thread::sleep_for (std::chrono::milliseconds(100));

    rs232Comm.shutdown();
}