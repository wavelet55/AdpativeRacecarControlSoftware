#include <iomanip> 
#include "vision_record_play.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace boost::filesystem;

namespace videre{
    
    VisionRecordPlay::VisionRecordPlay()
    {

        //Logger Setup
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);

        video_write_sptr_ = nullptr;
        framerate_ = 0;
        frame_width_ = 0;
        frame_height_ = 0;
        _videoWriterOpen = false;
    }

    VisionRecordPlay::~VisionRecordPlay()
    {
        ClearVideoWriter();
    }
    
    bool VisionRecordPlay::Init(string directory, string file_base,
                           int framerate, int frame_width, int frame_height){
        
        if(!is_directory(directory))
            create_directories(directory);
        
        directory_ = directory;
        file_base_name_ = file_base;
        video_number_ = 0;
        framerate_ = framerate;
        frame_width_ = frame_width;
        frame_height_ = frame_height; 
        
        SetCodec_CV_FOURCC('M', 'J', 'P', 'G');

        _videoWriterOpen = false;
        return true;
    }

    void VisionRecordPlay::SetImageParameters(int width, int height, int fps)
    {
        framerate_ = fps;
        frame_width_ = width;
        frame_height_ = height;
    }
    
    void VisionRecordPlay::SetCodec_CV_FOURCC(char c1, char c2, char c3, char c4){
        codec_ = cv::VideoWriter::fourcc(c1, c2, c3, c4);
    }
    
    void VisionRecordPlay::RecordFrameToVideo(cv::Mat& frame, TelemetryMessage &telemetry){
        if(video_write_sptr_ == nullptr)
            ResetVideoWriter();
            
        video_write_sptr_->write(frame); 
        metafile_.write(reinterpret_cast<char*>(&telemetry), sizeof(TelemetryMessage));
    }
    
    void VisionRecordPlay::ClearVideoWriter()
    {
        if( _videoWriterOpen ) {
            video_write_sptr_.reset();
            metafile_.close();
            _videoWriterOpen = false;
        }
    }

    bool VisionRecordPlay::SetVideoCodec(const std::string &codecName)
    {
        bool error = false;
        if(codecName.compare("XVID") == 0)
            SetCodec_CV_FOURCC('X', 'V', 'I', 'D');
        else if(codecName.compare("H264") == 0)
            SetCodec_CV_FOURCC('H','2','6','4');
        else if(codecName.compare("I420") == 0)
            SetCodec_CV_FOURCC('I','4','2','0');
        else
            SetCodec_CV_FOURCC('M', 'J', 'P', 'G');  //Default

        return error;
    }


    vector<path> VisionRecordPlay::ListVideosInDirectory(){
        
        recursive_directory_iterator it(directory_);
        recursive_directory_iterator endit;
        vector<path> filenames = vector<path>();

        while(it != endit)
        {
            if(is_regular_file(*it) && it->path().extension() == BINEXT) {
                filenames.push_back(it->path());
            }
            ++it;
        }
        
        return filenames;
    }
    
    void VisionRecordPlay::UnpackBinaryToCSV(){
        if(!exists(directory_) || !is_directory(directory_)) return;

        recursive_directory_iterator it(directory_);
        recursive_directory_iterator endit;
        vector<path> filenames = vector<path>();

        while(it != endit)
        {
            if(is_regular_file(*it) && it->path().extension() == BINEXT) {
                filenames.push_back(it->path());
            }
            ++it;
        }
        
        TelemetryMessage t = TelemetryMessage();
        for(int n = 0; n < filenames.size(); n++){
            std::ifstream in(filenames[n].c_str());
            filenames[n].replace_extension(CSVEXT);
            std::ofstream outc(filenames[n].c_str());
            outc << METAHEADER;
            stringstream stream;
             while(!in.eof()){
                in.read(reinterpret_cast<char*>(&t), sizeof(TelemetryMessage));
                stream.str(string());
                stream << To_String_P(t.IsFromHOPS) << ", " 
                     << To_String_P(t.LocalTimeSec) << ", " 
                     << To_String_P(t.DeltaTime) << ", " 
                     << To_String_P(t.CoordinatesLatLonOrXY) << ", " 
                     << To_String_P(t.LatitudeRadOrY) << ", " 
                     << To_String_P(t.LongitudeRadOrX) << ", " 
                     << To_String_P(t.AltitudeMSL) << ", " 
                     << To_String_P(t.HeightAGL) << ", " 
                     << To_String_P(t.VelEastMpS) << ", " 
                     << To_String_P(t.VelNorthMpS) << ", " 
                     << To_String_P(t.VelDownMpS) << ", " 
                     << To_String_P(t.RollRad) << ", " 
                     << To_String_P(t.PitchRad) << ", " 
                     << To_String_P(t.YawRad) << ", " 
                     << To_String_P(t.RollRateRadps) << ", " 
                     << To_String_P(t.PitchRateRadps) << ", " 
                     << To_String_P(t.YawRateRadps) << ", " 
                     << To_String_P(t.gpsTimeStampSec) << endl;
                outc << stream.str();  
             }          
            outc.close();
            in.close();
        }
    }
    
    /*------------------------------------------------------------------------*/
    
    void VisionRecordPlay::ResetVideoWriter(){
        if(_videoWriterOpen)
        {
            ClearVideoWriter();
        }
        try
        {
            video_number_++;
            auto filename = file_base_name_ +
                            to_string(video_number_) +
                            VIDEO_EXTENSION_STR;

            /* Create file for storing image frames */
            current_file_path_ = directory_ + "/" + filename;
            auto full_path = current_file_path_.c_str();
            video_write_sptr_.reset(new cv::VideoWriter(current_file_path_.c_str(),
                                                        codec_,
                                                        framerate_,
                                                        cv::Size(frame_width_,
                                                                 frame_height_),
                                                        true));


            /* Create meta file to store information about each frame */
            auto metafilename = file_base_name_ +
                                to_string(video_number_) + METATAG + BINEXT;
            current_meta_file_path_ = directory_ + "/" + metafilename;
            metafile_.open(current_meta_file_path_.c_str());
            _videoWriterOpen = true;
        }
        catch (std::exception &e)
        {
            LOGERROR("StreamRecordManager Record Image: Exception: " << e.what());
        }
    }
    
    string VisionRecordPlay::To_String_P(double d){
        stringstream stream;
        stream << fixed << setprecision(DOUBLEPRECISION) << d;
        return stream.str();   
    }
    
    
    
        
}
