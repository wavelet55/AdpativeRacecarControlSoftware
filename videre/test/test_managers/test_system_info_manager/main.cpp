
#include <iostream>
#include <RabitReactor.h>
#include "TestSystemInfoManager.h"
#include "system_info_manager.h"

using namespace std;
using namespace Rabit;
using namespace videre;

typedef std::unique_ptr<Rabit::RabitManager> ManagerPtr;
int main(int argc, char* argv[]) {

  std::cout << "***************************************************" << std::endl;
  std::cout << "*              Test SystemInfoManager             *" << std::endl;
  std::cout << "***************************************************" << std::endl;
  std::cout << std::endl;

  auto tm = ManagerPtr(new TestSystemInfoManager("TestSystemInfoManager"));
  auto sm = ManagerPtr(new SystemInfoManager("SystemInfoManager"));

  auto smptr = static_cast<SystemInfoManager*>(sm.get());
  smptr->Initialize();

  //cout << smptr->StaticInformation() << endl;

  auto reactor = Rabit::RabitReactor();

  reactor.AddManager(std::move(tm));
  reactor.AddManager(std::move(sm));

  reactor.Run();


}
