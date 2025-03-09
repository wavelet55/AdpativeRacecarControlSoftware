/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athrs: Randy Direen PhD, Harry Direen PhD
 * Date: July 2018
  *******************************************************************/
#ifndef VIDERE_DEV_DETECTGLYPHS_H
#define VIDERE_DEV_DETECTGLYPHS_H

#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "CommonImageProcTypesDefs.h"
#include "GlyphModel.h"


namespace CudaImageProcLibsTrackHeadNS
{

    struct GlyphAndOrient
    {
        GlyphAndOrient(int i, int r):glyph_idx(i), glyph_rot(r){}
        int glyph_idx;
        int glyph_rot; //0, 1, 2, or 3
    };

    class DetectGlyphs
    {

    public:
        DetectGlyphs();

        ~DetectGlyphs();


        /**
        * @brief
        */
        bool init();

        /**
         * @brief Finds all the gyphs within the frame
         *
         * @param frame
         */
        bool update(cv::Mat& frame);

        /**
        * @brief Finds all the gyphs within the frame
        *
        * @param frame
         *
         * @return true if ok, false if error.
        */
        bool update(cv::cuda::GpuMat& cframe);

        /**
         * Use this to load the model to track
         *
         * @param gm
         */
        void set_glyph_model(GlyphModel &gm);

        void set_canny_low_thresh(int val)
        {
            val = val < 0 ? 0 : val;
            val = val > 255 ? 255 : val;

            canny_low_thresh_ = val;
        }

        void set_canny_high_thresh(int val)
        {
            val = val < 0 ? 0 : val;
            val = val > 255 ? 255 : val;

            canny_high_thresh_ = val;
        }

        void set_canny_kernel(int val)
        {
            if(val != 3 || val != 5 || val != 7)
                val = 3;

            canny_kernel_ = val;
        }

        void set_max_glyph_area(int val){
            max_glyph_area_ = val;

        }

        void set_min_glyph_area(int val){
            min_glyph_area_ = val;

        }
        void set_glyph_thresh(float val){
            val = val < 0 ? 0 : val;
            val = val > 1 ? 1 : val;

            glyph_thresh_ = val;
        }

        /**
        * @brief Return square contours
        *
        * @return
        */
        std::vector<std::vector<cv::Point> > & contours(){return fcontours_;};

        /**
        * @brief
        *
        * @return
        */
        std::vector<cv::Vec4i>& hierarchy(){ return fhierarchy_;}

        /**
        * @brief Return squares corresponding do glyphs
        *
        * @return
        */
        std::vector<std::array<cv::Point, 4> >& squares(){return squares_;}

        /**
        * @brief The indices into the model
        *
        * @return
        */
        std::vector<int>& indices(){return indices_;}

        /**
        * @brief Returns the points corresponding to the square corners
        *
        * @return
        */
        std::vector<cv::Point2f>& points(){return points_;}


        /**
        * @brief Center of mass of the squares
        *
        * @return
        */
        std::vector<cv::Point2f>& center_mass(){return mc_;}

        /**
        * @brief
        *
        * @return
        */
        cv::Mat debugMat(){return boxmap_;}

        cv::Mat edge(){return edge_;}


    private:

        /**
         * returns true if ok, false if error;
         * @return
         */
        bool work();

        /**
        * @brief Find all the glyphs within an image
        */
        bool find_all_glyphs();



        /**
        * @brief Takes an image of a glyph and returns a probability vector
        *
        * @param img_bw
        * @param vec
        *
        * @return
        */
        float process_glyph(cv::Mat& img_bw, std::vector<float>& vec);

        /**
        * @brief Integrates over a glyph point and returns a probabilty of found
        *
        * @param img_bw
        * @param n
        * @param m
        * @param cellSize
        * @param offset
        * @param
        *
        * @return
        */
        float process_glyph_calc_intensity(cv::Mat& img_bw, size_t n, size_t m,
                                           int cellSize,
                                           int offset,
                                           int cellDetect );

        /**
        * @brief returns a vector of 1s and 0s given the glyph_vec
                *
                *   The glyph_vec is a kind of probability vector of the glyph points. With
        * this function you set a threshold that says you believe the glyph point
        * is balck or white.
        *
        * @param glyph_vec
        * @param threshold a value between 0 and 1
        */
        bool thresh_glyph_vec(std::vector<float>& glyph_vec, float threshold);

        /**
        * @brief Given a glyph pattern vec, we search through all the known
        * glyphs and see if we recognize one in the database.
        * The function returns the index to the glyph it thinks we are most likely
        * looking at.
        *
        * @param vec
        *
        * @return
        */
        GlyphAndOrient match_glyph(std::vector<float>& vec);


        /**
        * @brief Filter out multiple detections of the same glyph
        *
        *   When you get all the contours back, there are multipls squares that
        * encompase the glyph. This routing will filter out those extra squares
        * and keep the smallest one.
        */
        bool filter_final();

    public:
        bool UseGPU = false;
        ImageProcLibsNS::HeadTrackingImageDisplayType_e DisplayType;

    private:
        struct GlyphData{
            double area;
            std::vector<cv::Point2f> approx;
            int rotn = 0;

            void rotate_points_in_place(int n);

        };

        GlyphModel *glyph_model_ptr_ = nullptr;

        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        const size_t GlyphPixSize = 300;

        int canny_low_thresh_ = 50;
        int canny_high_thresh_ = 150;
        int canny_kernel_ = 3;

        float glyph_thresh_ = 0.7f;

        int max_glyph_area_ = 8000;
        int min_glyph_area_ = 1000;

        cv::Ptr<cv::cuda::CannyEdgeDetector> ccanny_;
        cv::Ptr<cv::cuda::Filter> cdilate_;

        cv::Mat gray_;
        cv::Mat edge_;
        cv::Mat thresh_;

        cv::cuda::GpuMat cgray_;
        cv::cuda::GpuMat cedge_;

        cv::Mat box1_;
        cv::Mat boxresized_;

        cv::Mat boxmap_;
        cv::Mat pointmap1_;
        cv::Mat pointmap2_;

        cv::Mat img_bw_;
        cv::Mat img_rgb_;


        std::vector<std::array<float, 25> > const_glyph_set_;

        std::vector<std::vector<cv::Point> > contours_;
        std::vector<cv::Vec4i> hierarchy_;

        std::vector<std::array<cv::Point, 4> > squares_;
        std::vector<cv::Point2f> approx_;
        std::array<cv::Point, 4> approxi_;

        std::vector<int> indices_;
        std::vector<cv::Point2f> points_;

        std::vector<std::vector<cv::Point> > fcontours_;
        std::vector<cv::Vec4i> fhierarchy_;
        std::vector<cv::Point2f> mc_;
        std::vector<cv::Point2f> dst_;
        cv::Mat trans_mat33_;
        std::vector<float> glyph_vec_;

        std::vector<std::vector<GlyphData> > glyph_data_;



    };

}


#endif //VIDERE_DEV_DETECTGLYPHS_H
