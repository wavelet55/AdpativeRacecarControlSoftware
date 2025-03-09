
#include <iostream>
#include "config_data.h"

using namespace std;
using namespace videre;

int main(int argc, char* argv[]) {

  std::cout << "***************************************************" << std::endl;
  std::cout << "*              Test ConfigData                    *" << std::endl;
  std::cout << "***************************************************" << std::endl;
  std::cout << std::endl;

  auto cd = ConfigData();
  cd.ParseConfigFile("config.ini.sample");

  cout << cd.GetByPath<bool>("flags.testing_without_images") << endl;
  cout << cd.GetByPath<bool>("flags.testing_without_sockets") << endl;
  cout << cd.GetByPath<bool>("flags.show_display_window") << endl;

  cout << cd.GetByPath<string>("zeromq.host_reply") << endl;
  cout << cd.GetByPath<string>("zeromq.host_pub_results") << endl;
  cout << cd.GetByPath<string>("zeromq.host_pub_video") << endl;
  cout << cd.GetByPath<string>("zeromq.host_sub_telemetry") << endl;
  cout << cd.GetByPath<string>("zeromq.host_pub_monitor") << endl;

  cout << cd.GetByPath<string>("opencv.device") << endl;
  cout << cd.GetByPath<double>("opencv.height") << endl;
  cout << cd.GetByPath<double>("opencv.width") << endl;
  cout << cd.GetByPath<double>("opencv.fps") << endl;

  cout << cd.GetByPath<string>("video_recording.directory") << endl;
  cout << cd.GetByPath<string>("video_recording.base_name") << endl;
  cout << cd.GetByPath<int>("video_recording.fps") << endl;

  cout << cd.GetByPath<int>("video_stream.quality") << endl;

}
