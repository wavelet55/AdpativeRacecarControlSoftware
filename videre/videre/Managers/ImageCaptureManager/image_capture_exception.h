#ifndef IMAGE_CAPTURE_EXCEPTION
#define IMAGE_CAPTURE_EXCEPTION

#include <string>
#include <exception>

namespace videre{

  class ImageCaptureException : public std::exception{

  private:
    std::string _message;

  public:

    ImageCaptureException(std::string msg){
      _message = msg;
    }

    virtual const char* what() const throw(){
      return _message.c_str();
    }
  };
}

#endif //IMAGE_CAPTURE_EXCEPTION
