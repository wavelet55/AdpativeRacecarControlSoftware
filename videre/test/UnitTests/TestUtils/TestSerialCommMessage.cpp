//
// Created by nvidia on 1/15/18.
//

#include "TestSerialCommMessage.h"

using namespace dtiUtils;
using namespace std;

TEST_F(TestSerialCommMessage, TestMsgOperation)
{
    bool tstFlg;
    int tstInt;
    u_char msgBuf[25];

    SerialCommMessage_t scMsg;

    scMsg.init(10);

    scMsg.clearMsg();

    string msg1 = "Tst=123";
    scMsg.addMsg(msg1);

    tstInt = scMsg.MsgSize;
    EXPECT_EQ(tstInt, 7);

    tstInt = scMsg.getMsg(msgBuf);
    EXPECT_EQ(tstInt, 7);
    EXPECT_EQ(msgBuf[3], '=');

    scMsg.addMsg(msgBuf, 4, 3, 7);
    tstInt = scMsg.MsgSize;
    EXPECT_EQ(tstInt, 10);
    tstInt = scMsg.getMsg(msgBuf);
    EXPECT_EQ(msgBuf[7], '=');

}