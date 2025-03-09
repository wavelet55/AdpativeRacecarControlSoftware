//
// Created by nvidia on 1/15/18.
//

#include "TestLockFreeQueue.h"
#include "SerialCommMessage.h"

#include <memory>
#include <string>

using namespace std;
using namespace dtiUtils;

TEST_F(TestLockFreeQueue, TestStringCaseInsensitiveCompare)
{
    int QueueSize = 5;
    int MaxMsgSize = 10;
    bool tstFlg;
    LockFreeQueue<SerialCommMessage_t> mq;
    mq.init(QueueSize);

    //Init message queue items
    for(int i = 0; i < QueueSize; i++)
    {
        mq.getQueuePtr()[i].init(MaxMsgSize);
    }

    tstFlg = mq.IsQueueEmpty();
    EXPECT_EQ(tstFlg, true);

    tstFlg = mq.IsQueueFull();
    EXPECT_EQ(tstFlg, false);

    //Add Message to queue
    string msg1 = "Msg1=123";
    mq.getHeadItemReference().addMsg(msg1);
    mq.IncrementHead();

    tstFlg = mq.IsQueueEmpty();
    EXPECT_EQ(tstFlg, false);

    int msgsize = mq.getTailItemReference().MsgSize;
    EXPECT_EQ(msgsize, 8);

    mq.IncrementTail();
    tstFlg = mq.IsQueueEmpty();
    EXPECT_EQ(tstFlg, true);

    for(int i = 0; i < QueueSize - 2; i++)
    {
        mq.IncrementHead();
        tstFlg = mq.IsQueueFull();
        EXPECT_EQ(tstFlg, false);
    }

    mq.IncrementHead();
    EXPECT_EQ(tstFlg, true);

    mq.IncrementHead();
    EXPECT_EQ(tstFlg, true);

    mq.IncrementTail();
    tstFlg = mq.IsQueueFull();
    EXPECT_EQ(tstFlg, false);

    mq.IncrementHead();

    for(int i = 0; i < QueueSize - 2; i++)
    {
        mq.IncrementTail();
        tstFlg = mq.IsQueueEmpty();
        EXPECT_EQ(tstFlg, false);
    }

    mq.IncrementTail();
    tstFlg = mq.IsQueueEmpty();
    EXPECT_EQ(tstFlg, true);

    mq.IncrementTail();
    tstFlg = mq.IsQueueEmpty();
    EXPECT_EQ(tstFlg, true);

}
