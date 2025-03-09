#include "gtest/gtest.h"
#include <log4cxx/logger.h>
#include <log4cxx/xml/domconfigurator.h>
#include <log4cxx/helpers/exception.h>
#include <iostream>
#include <memory>

using namespace log4cxx;
using namespace log4cxx::xml;

//Setup and run unit tests
int main(int argc, char **argv)
{
    std::string log_config_file = "Log4CxxConfig.xml";
    /* Setup the logger and print a message indicating that the program is
     starting */
    DOMConfigurator::configure(log_config_file);
    auto rootLogger = Logger::getRootLogger();

    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}