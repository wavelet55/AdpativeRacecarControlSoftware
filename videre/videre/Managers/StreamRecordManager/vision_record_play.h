/* ****************************************************************
 * Athr(s): Harry Direen PhD, Randy Direen Phd.
 * DireenTech Inc.  (www.DireenTech.com)
 * Date: March. 2016
 * 
 * Academy Center for UAS Research 
 * Department of Electrical and Computer Engineering 
 * HQ USAFA/DFEC 
 * 2354 Fairchild Drive 
 * USAF Academy, CO 80840-6236 
 * 
 *******************************************************************/

#ifndef VISION_RECORD_PLAY
#define	VISION_RECORD_PLAY

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "../../Messages/telemetry_message.h"
#include "../../Utils/logger.h"

namespace videre
{

    class VisionRecordPlay {

        //Logging System
        log4cxx::LoggerPtr log4cpp_;

    public:
        
        VisionRecordPlay();

        ~VisionRecordPlay();
        
        /**
         * Initialize.
         * @param directory to save video and data files to.
         * @param file_base base name of the individual files.
         * @param framerate 
         * @param frame_width
         * @param frame_height
         * @return 
         */
        bool Init(std::string directory, std::string file_base,
                  int framerate, int frame_width, int frame_height);
        
        
        /**
         * Sets the codec using CV_FOURCC.
         * (http://www.emgu.com/wiki/files/1.3.0.0/html/493ad812-fdce-ca3f-eb35-7e732468741e.htm)
         * 
         * @param c1
         * @param c2
         * @param c3
         * @param c4
         */
        void SetCodec_CV_FOURCC(char c1, char c2, char c3, char c4);
        
        /**
         * Seems to take about 5-8ms to compress an image on my machine.
         */
        void SetCodec_MJPG(){
            SetCodec_CV_FOURCC('M', 'J', 'P', 'G');
        }
        
        void SetCodec_XVID(){
            SetCodec_CV_FOURCC('X', 'V', 'I', 'D');
        }
        
        /**
         * Keeps up with framerate but the files grow really fast, about a gig
         * every 20 sec at 1080p 30 fps.
         */
        void SetCodec_I420(){
            SetCodec_CV_FOURCC('I','4','2','0');
        }
        
        void SetCodec_H264(){
            SetCodec_CV_FOURCC('X','2','6','4');
        }

        void SetImageParameters(int width, int height, int fps);

        bool IsOpen() {return _videoWriterOpen; }
        
        /**
         * Record a single frame to file. If the file isn't already open, this
         * routine will open it. Call ClearVideoWriter() to start a new file.
         * @param frame is the frame being written to file.
         * @param telemetry is the vehicle telemetry information.
         */
        void RecordFrameToVideo(cv::Mat& frame, TelemetryMessage& telemetry);
        
        bool SetVideoCodec(const std::string &codecName);

        /**
         * Closes out the current file being written too.
         */
        void ClearVideoWriter();
        
        /**
         * Used by another program to take binary files and convert them to 
         * csv. This is so storage can be done as quickly as possible during a
         * mission, and when landed, it can convert to larger files.
         */
        void UnpackBinaryToCSV();
        
        /**
         * List video files that have been created in flight.
         * @return 
         */
        std::vector<boost::filesystem::path> ListVideosInDirectory();
        

        /**
         * Resets the VideoWriter for a new file.
         */
        void ResetVideoWriter();
        
        std::string To_String_P(double d);
        
    public:
        std::string comma_extension(){
            return CSVEXT;
        }
    
    private:
        
        const std::string VIDEO_EXTENSION_STR = ".avi";
        const std::string BINEXT = ".dat";
        const std::string CSVEXT = ".csv";
        const std::string METATAG = "_meta";
        const std::string METAHEADER = "IsFromHOPS, LocalTimeSec, DeltaTime, LatLonOrXY, "
                                       "LatOrY, LatOrX, Alt, "
                                       "Height, VelEMpS, VelNMps, VelDMpS, "
                                       "Roll, Pitch, Yaw, RollR, PitchR, YawR, "
                                       "gpsTimeSec \n";
        
        const int DOUBLEPRECISION = 14;
        int codec_;   /* Set this with CV_FOURCC */
        
        std::shared_ptr<cv::VideoWriter> video_write_sptr_; /* record images to here */
        
        std::string directory_; /* Directory that holds videos. */
        std::string file_base_name_; /* Base filename of videos. */
        boost::filesystem::path current_file_path_; /* Full path to current video */
        
        std::ofstream metafile_; /* store meta data to here */
        boost::filesystem::path current_meta_file_path_; /* Full path to meta data */
        
        int video_number_; /* This will be appended to the file_base_name */
        
        int framerate_; /* Framerate to set the avi container at */
        int frame_width_;
        int frame_height_;

        std::string _videoCodec = "MJPEG";

        bool _videoWriterOpen = false;

    };

}


#endif	/* VISION_RECORD_PLAY */

