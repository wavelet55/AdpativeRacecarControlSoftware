#ifndef TEST_SYSTEM_INFO_MANAGER
#define TEST_SYSTEM_INFO_MANAGER

#include <iostream>
#include <string>
#include <RabitManager.h>
#include "sysinfo_dynamic_message.h"
#include "sysinfo_static_message.h"

class TestSystemInfoManager : public Rabit::RabitManager {

private:
    int count = 0;

    std::shared_ptr<SysInfoDynamicMessage> _sysinfo_dynamic_msg;
    std::shared_ptr<SysInfoStaticMessage> _sysinfo_static_msg;


public:
    TestSystemInfoManager(std::string name ) : RabitManager(name){
        this->SetWakeupTimeDelayMSec(100);

	_sysinfo_dynamic_msg = std::make_shared<SysInfoDynamicMessage>("SysInfoDynamicMessage");
	this->AddPublishSubscribeMessage(_sysinfo_dynamic_msg->GetMessageTypeName(), _sysinfo_dynamic_msg);

	_sysinfo_static_msg = std::make_shared<SysInfoStaticMessage>("SysInfoStaticMessage");
	this->AddPublishSubscribeMessage(_sysinfo_static_msg->GetMessageTypeName(), _sysinfo_static_msg);
    }

    void ExecuteUnitOfWork() final {
        count++;
        if(count > 100)
            this->ShutdownAllManagers(true);

	if(count == 1) {
	  if(_sysinfo_static_msg->FetchMessage())
	    std::cout << _sysinfo_static_msg->msg_str << std::endl;
	}

	if(_sysinfo_dynamic_msg->FetchMessage())
	  std::cout << _sysinfo_dynamic_msg->msg_str << std::endl;


    }

};

#endif //TEST_SYSTEM_INFO_MANAGER
