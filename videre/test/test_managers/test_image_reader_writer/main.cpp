
#include <iostream>
#include <memory>
#include <exception>
#include <RabitReactor.h>
#include "config_data.h"
#include "comms_manager.h"
#include "ImagePlusMetadataRecorder.h"
#include "ImagePlusMetadataReader.h"
#include "ImagePlusMetadataFileHeaders.h"
#include "image_plus_metadata_message.h"
#include "CompressedImageMessage.h"


using namespace std;
using namespace Rabit;
using namespace videre;
using namespace VidereImageprocessing;

typedef std::unique_ptr<Rabit::RabitManager> ManagerPtr;

int main(int argc, char *argv[])
{
    ImageReturnType_e irt;
    int imgCounter = 0;

    std::cout << "***************************************************" << std::endl;
    std::cout << "*              Test Image Reader/Writer                  *" << std::endl;
    std::cout << "***************************************************" << std::endl;
    std::cout << std::endl;

    auto config_sptr = make_shared<ConfigData>();

    config_sptr->ParseConfigFile("VidereConfig.ini");
    string imgDirectory = "ImagePMDFiles";

    ImagePlusMetadataMessage ipmMsg;
    CompressedImageMessage compImage;

    ImagePlusMetadataReader ImgReader(config_sptr);


    try
    {
        int NoImgFiles = ImgReader.GetListOfFilesFromDirectory(imgDirectory);
        cout << "Number of Files = " << NoImgFiles << endl;

        while(true)
        {
            irt = ImgReader.ReadNextImagePlusMetadata(&ipmMsg, &compImage);
            cout << "Image Return Type = " << irt << endl;
            if(irt == ImageReturnType_e::IRT_EndOfImages
                    || irt == ImageReturnType_e::IRT_Error)
            {
                cout << "End of Images" << endl;
                break;
            }
            else
            {
                ++imgCounter;
                cout << "Image Counter: " << imgCounter << endl;
                cout << "Image Number: " << ipmMsg.ImageNumber << endl;
                cout << "Image Size: " << compImage.ImageBuffer.size() << endl;
                cout << endl;
            }
        }

        ImgReader.closeImageFile();
    }
    catch (exception &e)
    {
        cout << e.what() << endl;
    }

    cout <<  "Done Testing"  << endl;
}
