/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athrs: Randy Direen PhD, Harry Direen PhD
 * Date: July 2018
  *******************************************************************/

#include "DetectGlyphs.h"
#include <algorithm>
#include <iostream>
#include "opencv2/cudalegacy.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "MathUtils.h"

using namespace ImageProcLibsNS;

namespace CudaImageProcLibsTrackHeadNS
{

#define PRINTPOINT(idx, x, y) \
         putText(pointmap1_, cv::format("%d", idx), \
                 cv::Point(x - 5, y - 15), \
                 cv::FONT_HERSHEY_PLAIN, 1.5, \
                 cv::Scalar(0,255,0), 2);

    DetectGlyphs::DetectGlyphs()
    {
        log4cpp_ = log4cxx::Logger::getLogger("aobj");
        log4cpp_->setAdditivity(false);
    }

    DetectGlyphs::~DetectGlyphs()
    {}


    bool DetectGlyphs::init()
    {
        bool error = false;

        if(glyph_model_ptr_ == nullptr)
        {
            LOGERROR("TrackHeadProcess:TrackHeadPostion: Need to set a model before running");
            error = true;
            return error;
        }

        try
        {
            ccanny_ = cv::cuda::createCannyEdgeDetector(canny_low_thresh_,
                                                        canny_high_thresh_,
                                                        canny_kernel_);

            cdilate_ = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE,
                                                        CV_8UC1, cv::Mat());

            box1_ = cv::Mat::zeros(GlyphPixSize, GlyphPixSize, CV_8UC3);
            boxresized_ = cv::Mat::zeros(GlyphPixSize, GlyphPixSize, CV_8UC3);
            boxmap_ = cv::Mat::zeros(600, 900, CV_8UC3);
            dst_.resize(4);
            glyph_vec_.resize(25);
            glyph_data_.resize(glyph_model_ptr_->num_glyphs());
        }
        catch (std::exception &e)
        {
            LOGERROR("TrackHeadProcess:TrackHeadPostion: Exception: " << e.what());
            error = true;
        }
        return error;
    }

    void DetectGlyphs::set_glyph_model(GlyphModel &gm)
    {
        glyph_model_ptr_ = &gm;
    }

    /**
    * @brief Finds all the gyphs within the frame
    *
    * @param frame
    */
    bool DetectGlyphs::update(cv::Mat& frame)
    {
        bool ok = true;
        try
        {
            gray_ = frame;

            Canny(gray_, edge_, canny_low_thresh_, canny_high_thresh_, canny_kernel_);

            dilate(edge_, edge_, cv::Mat(), cv::Point(-1, -1), 1);

            ok = work();
        }
        catch (std::exception &e)
        {
            LOGERROR("DetectGlyphs:update: Exception: " << e.what());
            ok = false;
        }
        return ok;
    }


    /**
    * @brief Finds all the gyphs within the frame
    *
    * @param frame
    */
    bool DetectGlyphs::update(cv::cuda::GpuMat& cframe)
    {
        bool ok = true;
        try
        {
            cgray_ = cframe;

            cgray_.download(gray_);

            ccanny_->detect(cgray_, cedge_);

            cdilate_->apply(cedge_, cedge_);

            cedge_.download(edge_);

            ok = work();
        }
        catch (std::exception &e)
        {
            LOGERROR("DetectGlyphs:update: Exception: " << e.what());
            ok = false;
        }
        return ok;
    }


    bool DetectGlyphs::work()
    {
        bool ok = false;
        try
        {
            contours_.clear();
            hierarchy_.clear();

            findContours(edge_, contours_, hierarchy_,
                         cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));


            squares_.clear();
            indices_.clear();
            points_.clear();

            fcontours_.clear();
            fhierarchy_.clear();
            mc_.clear();

            for(int i = 0; i < glyph_data_.size(); ++i)
                glyph_data_[i].clear();

            find_all_glyphs();

            // result is placed in squares_
            filter_final();

            ok = true;
        }
        catch (std::exception &e)
        {
            LOGERROR("DetectGlyphs:work: Exception: " << e.what());
            ok = false;
        }
        return ok;
    }

    /**
    * @brief Find all the glyphs within an image
    *****/
    bool DetectGlyphs::find_all_glyphs()
    {
        bool error = false;
        try
        {
            size_t ccount = 0;

            if(DisplayType == HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector)
            {
                boxmap_ = cv::Scalar::all(0);
                putText(boxmap_, "Redundant Detected Glyphs: ",
                        cv::Point(10,20),
                        cv::FONT_HERSHEY_PLAIN, 1.0,
                        cv::Scalar(0,255,0), 1);

                putText(boxmap_, "Detected Glyphs: ",
                        cv::Point(10,150),
                        cv::FONT_HERSHEY_PLAIN, 1.0,
                        cv::Scalar(0,255,0), 1);
            }

            for(int i = 0; i < contours_.size(); ++i)
            {
                approxPolyDP(cv::Mat(contours_[i]), approx_,
                             cv::arcLength(cv::Mat(contours_[i]), true)*0.02,
                             true);

                cv::Moments ms = cv::moments(approx_);

                double area = cv::contourArea(approx_);
                if(area > min_glyph_area_ && area < max_glyph_area_ &&
                   cv::isContourConvex(cv::Mat(approx_)) &&
                   approx_.size() == 4)
                {

                    fcontours_.push_back(contours_[i]);
                    fhierarchy_.push_back(hierarchy_[i]);

                    mc_.push_back(cv::Point2f(ms.m10/ms.m00, ms.m01/ms.m00));

                    MathLibsNS::order_4_points(approx_);
                    cv::Point mwh = MathLibsNS::max_w_and_h(approx_);
                    dst_[0] = cv::Point2f(0, 0);
                    dst_[1] = cv::Point2f(mwh.x - 1, 0);
                    dst_[2] = cv::Point2f(mwh.x - 1, mwh.y - 1);
                    dst_[3] = cv::Point2f(0, mwh.y - 1);

                    trans_mat33_ = cv::getPerspectiveTransform(approx_, dst_);
                    cv::warpPerspective(gray_, box1_, trans_mat33_, mwh);
                    cv::resize(box1_, boxresized_, cv::Point(100, 100));
                    cv::threshold(boxresized_, img_bw_, 0, 255,
                                  cv::THRESH_BINARY | cv::THRESH_OTSU);

                    float conf = process_glyph(img_bw_, glyph_vec_);

                    if(conf > 0.3f){

                        thresh_glyph_vec(glyph_vec_, glyph_thresh_);

                        auto glyph_orient = match_glyph(glyph_vec_);
                        if(glyph_orient.glyph_idx >= 0)
                        {
                            GlyphData gld;
                            gld.area = area;
                            gld.approx = approx_;
                            gld.rotn = glyph_orient.glyph_rot;
                            glyph_data_[glyph_orient.glyph_idx].push_back(gld);

                            if(DisplayType == HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector
                                    && ccount < 8)
                            {

                                cvtColor(img_bw_, img_rgb_, cv::COLOR_GRAY2BGR);
                                img_rgb_.copyTo(boxmap_(
                                        cv::Rect(cv::Point(ccount*100,30),
                                                 boxresized_.size())));
                            }
                            ccount++;

                        }
                    }
                    else {
                        ccount = 0;
                    }
                }
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("DetectGlyphs:find_all_glyphs: Exception: " << e.what());
            error = true;
        }
        return error;
    }

    /**
    * @brief Takes an image of a glyph and returns a probability vector
    *
    * @param img_bw
    * @param vec
    *
    * @return
    */
    float DetectGlyphs::process_glyph(cv::Mat& img_bw, std::vector<float>& vec)
    {
        float confidence = 1.0f;
        try
        {
            const int glyphN = 5;
            int cellSize = img_bw_.cols / glyphN;

            int offset = (int) (cellSize * 0.2);

            int cellDetect = (int) (cellSize * 0.6);

            float cellDetectA = (float) (cellDetect * cellDetect);

            float intens = 0;
            for(size_t n = 0; n < glyphN; ++n)
            {
                for(size_t m = 0; m < glyphN; ++m)
                {

                    intens = process_glyph_calc_intensity(img_bw_, n, m,
                                                          cellSize,
                                                          offset,
                                                          cellDetect
                    );
                    vec[n * glyphN + m] = intens / cellDetectA;
                    float conf = fabs(vec[n * glyphN + m] - 0.5f) + 0.5f;
                    if(conf < confidence)
                        confidence = conf;
                }
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("DetectGlyphs:process_glyph: Exception: " << e.what());
        }
        return confidence;
    }

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
    float DetectGlyphs::process_glyph_calc_intensity(cv::Mat& img_bw, size_t n, size_t m,
                                       int cellSize,
                                       int offset,
                                       int cellDetect )
    {
        float intensity = 0;
        try
        {
            for(int i = 0; i < cellDetect; ++i)
            {
                for(int j = 0; j < cellDetect; ++j)
                {
                    intensity += ((float) img_bw_.at<unsigned char>(
                            n * cellSize + offset + i,
                            m * cellSize + offset + j)) / 255.0;
                }
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("DetectGlyphs:process_glyph_calc_intensity: Exception: " << e.what());
        }
        return intensity;
    }

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
    bool DetectGlyphs::thresh_glyph_vec(std::vector<float>& glyph_vec, float threshold)
    {
        bool error = false;
        try
        {
            for(int i = 0; i < glyph_vec.size(); ++i)
            {
                if(glyph_vec[i] > threshold)
                    glyph_vec[i] = 1.0f;
                else
                    glyph_vec[i] = 0;
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("DetectGlyphs:thresh_glyph_vec: Exception: " << e.what());
            error = true;
        }
        return error;
    }

    /**
    * @brief Given a glyph pattern vec, we search through all the known
    * glyphs and see if we recognize one in the database. The res
    * vector will show how many matches have been found within the particular
    * glyph. The function returns the glyph it thinks we are most likely
    * looking at.
    *
    * @param vec
    * @param res
    *
    * @return
    */
    GlyphAndOrient DetectGlyphs::match_glyph(std::vector<float>& vec)
    {
        int glyph_idx = -1;
        int glyph_rot = -1;

        std::array<uint8_t, PSIZE> uvec;
        for(size_t i = 0; i < PSIZE; i++)
            uvec[i] = static_cast<uint8_t>(vec[i]);

        Glyph g_in(uvec);

        size_t n_glyphs = glyph_model_ptr_->num_glyphs();

        try
        {
            for(size_t gl_idx = 0; gl_idx < n_glyphs; ++gl_idx)
            {
                Glyph g = glyph_model_ptr_->get_glyph_by_index(gl_idx);

                if(dotf(g_in, g) == PSIZE)
                {
                    glyph_idx = gl_idx; glyph_rot = 0;
                    break;
                }

                if(dotf(g_in, rotate_n_times(g, 1)) == PSIZE)
                {
                    glyph_idx = gl_idx; glyph_rot = 1;
                    break;
                }

                if(dotf(g_in, rotate_n_times(g, 2)) == PSIZE)
                {
                    glyph_idx = gl_idx; glyph_rot = 2;
                    break;
                }

                if(dotf(g_in, rotate_n_times(g, 3)) == PSIZE)
                {
                    glyph_idx = gl_idx; glyph_rot = 3;
                    break;
                }
            }

        }
        catch (std::exception &e)
        {
            LOGERROR("DetectGlyphs:match_glyph: Exception: " << e.what());
            glyph_idx = -1;
        }

        return GlyphAndOrient(glyph_idx, glyph_rot);
    }


    /**
    * @brief Filter out multiple detections of the same glyph
    *
    *   When you get all the contours back, there are multiple squares that
    * encompass the glyph. This routing will filter out those extra squares
    * and keep the smallest one.
    */
    bool DetectGlyphs::filter_final()
    {
        bool error = false;
        try
        {
            for(int i = 0; i < glyph_data_.size(); ++i)
            {
                int min_area = INT_MAX;
                int idx = 0;
                if(glyph_data_[i].size() != 0)
                {
                    for(int j = 0; j < glyph_data_[i].size(); ++j)
                    {
                        if(glyph_data_[i][j].area < min_area)
                        {
                            min_area = glyph_data_[i][j].area;
                            idx = j;
                        }

                    }
                    double gdat = glyph_data_[i][idx].area;

                    approxi_[0] = cv::Point((int) glyph_data_[i][idx].approx[0].x,
                                            (int) glyph_data_[i][idx].approx[0].y);
                    approxi_[1] = cv::Point((int) glyph_data_[i][idx].approx[1].x,
                                            (int) glyph_data_[i][idx].approx[1].y);
                    approxi_[2] = cv::Point((int) glyph_data_[i][idx].approx[2].x,
                                            (int) glyph_data_[i][idx].approx[2].y);
                    approxi_[3] = cv::Point((int) glyph_data_[i][idx].approx[3].x,
                                            (int) glyph_data_[i][idx].approx[3].y);
                    squares_.push_back(approxi_);

                    for(int l = 0; l < 4; ++l)
                    {
                        int rot = glyph_data_[i][idx].rotn;
                        indices_.push_back(i*4 + ((l - rot)%4 + 4)%4 );
                        points_.push_back(glyph_data_[i][idx].approx[l]);
                    }

                    if(DisplayType == HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector)
                    {
                        putText(boxmap_, cv::format("G[%d]", i),
                                cv::Point(10 + i * 100, 200),
                                cv::FONT_HERSHEY_PLAIN, 1.5,
                                cv::Scalar(0, 255, 0), 2);

                    }
                } else if(DisplayType == HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector)
                {
                    putText(boxmap_, cv::format("G[%d]", i),
                            cv::Point(10 + i * 100, 200),
                            cv::FONT_HERSHEY_PLAIN, 1.5,
                            cv::Scalar(0, 0, 255), 1);
                }
            }

            if(DisplayType == HeadTrackingImageDisplayType_e::HTID_HeadOrientationVector)
            {
                cvtColor(gray_, pointmap1_, cv::COLOR_GRAY2BGR);

                for(size_t i = 0; i < indices_.size(); ++i)
                {
                    PRINTPOINT(indices_[i], points_[i].x, points_[i].y)

                    circle(pointmap1_, points_[i], 2, cv::Scalar(0, 0, 255), 2, 7, 0);

                }
                cv::resize(pointmap1_, pointmap2_, cv::Size(), 0.60, 0.60);

                pointmap2_.copyTo(boxmap_(
                        cv::Rect(cv::Point(10, 230),
                                 pointmap2_.size())));
            }
        }
        catch (std::exception &e)
        {
            LOGERROR("DetectGlyphs:filter_final: Exception: " << e.what());
            error = true;
        }
        return error;
    }


    void DetectGlyphs::GlyphData::rotate_points_in_place(int n)
    {
        for(int k = 0; k < n; ++k)
            std::rotate(approx.rbegin(), approx.rbegin() + 1, approx.rend());
    }
}

