#ifndef CONFIG_PARSE_EXCEPTION
#define CONFIG_PARSE_EXCEPTION

#include <string>
#include <exception>

namespace videre{

  class ConfigParseException : public std::exception{

  private:
    std::string _message;

  public:

    ConfigParseException(){
      _message = "Cannot parse particular path of config file.";
    }

    ConfigParseException(std::string msg){
      _message = msg;
    }

    virtual const char* what() const throw(){
      return _message.c_str();
    }
  };
}

#endif //CONFIG_PARSE_EXCEPTION
