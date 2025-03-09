#include "testMessagePool.h"
#include <memory>
#include <string>

using namespace std;

TEST_F(TestMessagePool, TestMessagePoolConstructor)
{
    MessagePool* msgPoolPtr;

    //Create a Image Message Pool to test and play with.
    ImagePlusMetadataMessage imgMsg;
    msgPoolPtr = new MessagePool(&imgMsg, 5);

    int numberOfMsgs = msgPoolPtr->GetPoolSize();
    EXPECT_EQ(5, numberOfMsgs);

    numberOfMsgs = msgPoolPtr->GetNumberOfFreeMessages();
    EXPECT_EQ(5, numberOfMsgs);

    numberOfMsgs = msgPoolPtr->GetNumberOfCheckedOutMessages();
    EXPECT_EQ(0, numberOfMsgs);

    RabitMessage* msg1_ptr = msgPoolPtr->CheckOutMessage();
    EXPECT_NE(nullptr, msg1_ptr);

    EXPECT_EQ(imgMsg.GetTypeIndex(), msg1_ptr->GetTypeIndex());

    //Ensure messages are unique
    for(int i = 0; i < 4; i++)
    {
        RabitMessage* msg2_ptr = msgPoolPtr->CheckOutMessage();
        EXPECT_NE(msg1_ptr, msg2_ptr);
        msg1_ptr = msg2_ptr;

        numberOfMsgs = msgPoolPtr->GetNumberOfFreeMessages();
        EXPECT_EQ(3 - i, numberOfMsgs);

        numberOfMsgs = msgPoolPtr->GetNumberOfCheckedOutMessages();
        EXPECT_EQ(2 + i, numberOfMsgs);
    }
    delete msgPoolPtr;
}

TEST_F(TestMessagePool, TestMessagePoolCheckOutIn)
{
    ImagePlusMetadataMessage imgMsg;

    MessagePool msgPool(&imgMsg, 3);

    int numberOfMsgs = msgPool.GetPoolSize();
    EXPECT_EQ(3, numberOfMsgs);

    numberOfMsgs = msgPool.GetNumberOfFreeMessages();
    EXPECT_EQ(3, numberOfMsgs);

    numberOfMsgs = msgPool.GetNumberOfCheckedOutMessages();
    EXPECT_EQ(0, numberOfMsgs);

    RabitMessage* msg1_ptr = msgPool.CheckOutMessage();
    EXPECT_NE(nullptr, msg1_ptr);

    numberOfMsgs = msgPool.GetNumberOfFreeMessages();
    EXPECT_EQ(2, numberOfMsgs);

    numberOfMsgs = msgPool.GetNumberOfCheckedOutMessages();
    EXPECT_EQ(1, numberOfMsgs);

    numberOfMsgs = msgPool.findNumberOfFreeMessages();
    EXPECT_EQ(2, numberOfMsgs);

    int error = msgPool.CheckInMessage(msg1_ptr);
    EXPECT_EQ(0, error);

    //A second checkin of the same message should cause an error.
    error = msgPool.CheckInMessage(msg1_ptr);
    EXPECT_EQ(1, error);


    numberOfMsgs = msgPool.GetNumberOfFreeMessages();
    EXPECT_EQ(3, numberOfMsgs);

    numberOfMsgs = msgPool.GetNumberOfCheckedOutMessages();
    EXPECT_EQ(0, numberOfMsgs);


    msg1_ptr = msgPool.CheckOutMessage();
    RabitMessage* msg2_ptr = msgPool.CheckOutMessage();
    EXPECT_NE(nullptr, msg2_ptr);

    numberOfMsgs = msgPool.GetNumberOfFreeMessages();
    EXPECT_EQ(1, numberOfMsgs);

    numberOfMsgs = msgPool.GetNumberOfCheckedOutMessages();
    EXPECT_EQ(2, numberOfMsgs);

    RabitMessage* msg3_ptr = msgPool.CheckOutMessage();
    EXPECT_NE(nullptr, msg3_ptr);

    numberOfMsgs = msgPool.GetNumberOfFreeMessages();
    EXPECT_EQ(0, numberOfMsgs);

    numberOfMsgs = msgPool.GetNumberOfCheckedOutMessages();
    EXPECT_EQ(3, numberOfMsgs);

    //An attempt to check out a message when the pool is empty
    //should return a null pointer.
    RabitMessage* msg4_ptr = msgPool.CheckOutMessage();
    EXPECT_EQ(nullptr, msg4_ptr);

    error = msgPool.CheckInMessage(msg2_ptr);
    EXPECT_EQ(0, error);

    numberOfMsgs = msgPool.GetNumberOfFreeMessages();
    EXPECT_EQ(1, numberOfMsgs);

    numberOfMsgs = msgPool.GetNumberOfCheckedOutMessages();
    EXPECT_EQ(2, numberOfMsgs);

    error = msgPool.CheckInMessage(msg1_ptr);
    EXPECT_EQ(0, error);

    numberOfMsgs = msgPool.GetNumberOfFreeMessages();
    EXPECT_EQ(2, numberOfMsgs);

    numberOfMsgs = msgPool.GetNumberOfCheckedOutMessages();
    EXPECT_EQ(1, numberOfMsgs);

    msg2_ptr = msgPool.CheckOutMessage();
    EXPECT_NE(nullptr, msg2_ptr);

    numberOfMsgs = msgPool.GetNumberOfFreeMessages();
    EXPECT_EQ(1, numberOfMsgs);

    numberOfMsgs = msgPool.GetNumberOfCheckedOutMessages();
    EXPECT_EQ(2, numberOfMsgs);
}