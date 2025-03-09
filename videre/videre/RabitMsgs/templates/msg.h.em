
@###############################################
@#
@# Message generation for C++
@#
@# EmPy template for generating <msg>.cpp
@#
@###############################################
@# Start of Template
@###############################################

/* ****************************************************************
  *******************************************************************/

/* Auto-generated from file @file_name_in, do not modify directly */

@{
import genmsg.msgs
from rabit_msg_helper import *

msg_name = spec.short_name
}@

#pragma once

#include <string>
#include <memory>
#include <RabitMessage.h>

@{
################################################################
#                    Function Templates
################################################################
string_copy_template = """        void CopyTo{}(std::string s);

        void CopyTo{}(char* c, int L);"""
################################################################
# python functions for filling in parts of the template
################################################################
def list_variables(fields):
    for field in fields:
        if (field.is_builtin):
            if (not field.is_array):
                if field.type == "string":
                    print("        std::" + str(type_map[field.type]) + " " + str(field.name) + ";")
                else:
                    print("        " + str(type_map[field.type]) + " " + str(field.name) + ";")
            else:
                print("        std::array<{},{}> {};".format(str(type_map[field.base_type]), str(field.array_len), str(field.name)) )
        else:
            print("not here")

def get_set_variables(fields):
    for field in fields:
            if (field.is_builtin):
                if (not field.is_array):
                    if field.type == "string":
                        pass
                    else:
                        pass
                else:
                    print(string_copy_template.format(str(field.name), str(field.name)))
            else:
                    print("not here")
}@
namespace @(nm_space)
{
    class @(msg_name) : public Rabit::RabitMessage{

    public:
@list_variables(spec.parsed_fields())
    public:

        @(msg_name)();

        virtual std::unique_ptr<Rabit::RabitMessage> Clone() const final;

        virtual bool CopyMessage(Rabit::RabitMessage* msg) final;

        virtual void Clear() final;

@get_set_variables(spec.parsed_fields())
        virtual std::string ToString() const final;

        virtual int Serialize(uint8_t *buf, int maxBufSize, int stype = 0)  final;

        virtual int DeSerialize(uint8_t *buf, int len, int stype = 0) final;
    };

}
