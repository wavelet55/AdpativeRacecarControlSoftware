
#include <iostream>
#include <memory>
#include <exception>
#include <RabitReactor.h>
#include "config_data.h"
#include "TestCommsManager.h"
#include "comms_manager.h"

using namespace std;
using namespace Rabit;
using namespace videre;

typedef std::unique_ptr<Rabit::RabitManager> ManagerPtr;
int main(int argc, char* argv[]) {

    std::cout << "***************************************************" << std::endl;
    std::cout << "*              Test CommsManager                  *" << std::endl;
    std::cout << "***************************************************" << std::endl;
    std::cout << std::endl;

    auto config_sptr = make_shared<ConfigData>();

    config_sptr->ParseConfigFile("config.ini.sample");

    auto tm = ManagerPtr(new TestCommsManager("TestCommsManager"));
    auto cm = ManagerPtr(new CommsManager("CommsManager", config_sptr));

    try{
        static_cast<CommsManager*>(cm.get())->Initialize();
    }catch(exception& e) {
        cout << e.what() << endl;
    }

    auto reactor = Rabit::RabitReactor();

    reactor.AddManager(std::move(tm));
    reactor.AddManager(std::move(cm));

    reactor.Run();

}
