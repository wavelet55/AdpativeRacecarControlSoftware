
#include <iostream>
#include <memory>
#include <exception>
#include <RabitReactor.h>
#include "config_data.h"
#include "TestImageManager.h"
#include "image_capture_manager.h"
#include <log4cxx/logger.h>
#include <log4cxx/xml/domconfigurator.h>
#include <log4cxx/helpers/exception.h>

using namespace std;
using namespace Rabit;
using namespace videre;
using namespace log4cxx;
using namespace log4cxx::xml;

typedef std::unique_ptr<Rabit::RabitManager> ManagerPtr;
int main(int argc, char* argv[]) {

    std::cout << "***************************************************" << std::endl;
    std::cout << "*              Test ImageCaptureManager                  *" << std::endl;
    std::cout << "***************************************************" << std::endl;
    std::cout << std::endl;


    /* Setup the logger and print a message indicating that the program is
     starting */
    DOMConfigurator::configure("Log4CxxConfig.xml");
    auto rootLogger = Logger::getRootLogger();
    LOG4CXX_INFO(rootLogger, "Start Test Program");


    auto config_sptr = make_shared<ConfigData>();

    config_sptr->ParseConfigFile("VidereConfig.ini");

    auto tm = ManagerPtr(new TestImageManager("TestImageManager"));
    auto im = ManagerPtr(new ImageCaptureManager("ImageCaptureManager", config_sptr));

    try{
        static_cast<ImageCaptureManager*>(im.get())->Initialize();
    }catch(exception& e) {
        cout << e.what() << endl;
        return 1;
    }

    auto reactor = Rabit::RabitReactor();

    reactor.AddManager(std::move(tm));
    reactor.AddManager(std::move(im));

    reactor.Run();

}
