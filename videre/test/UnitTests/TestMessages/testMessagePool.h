#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <RabitManager.h>
#include "all_manager_message.h"
#include "video_control_message.h"
#include "video_process_message.h"
#include "image_plus_metadata_message.h"
#include "message_pool.h"

using namespace videre;
using namespace Rabit;

// The fixture for testing class Rabit.
class TestMessagePool : public ::testing::Test
{

protected:

    // You can do set-up work for each test here.
    TestMessagePool()
    {}

    // You can do clean-up work that doesn't throw exceptions here.
    virtual ~TestMessagePool()
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

