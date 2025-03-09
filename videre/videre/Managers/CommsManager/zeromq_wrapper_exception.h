#ifndef ZEROMQ_WRAPPER_EXCEPTION
#define ZEROMQ_WRAPPER_EXCEPTION

#include <string>
#include <exception>

namespace videre{

  class ZeroMQWrapperException : public std::exception{

  private:
    std::string _message;

  public:

    ZeroMQWrapperException(std::string msg){
      _message = msg;
    }

    virtual const char* what() const throw(){
      return _message.c_str();
    }
  };
}

#endif //ZEROMQ_WRAPPER_EXCEPTION
