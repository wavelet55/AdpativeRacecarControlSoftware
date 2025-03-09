/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athrs: Randy Direen PhD, Harry Direen PhD
 * Date: August 2018
  *******************************************************************/
#ifndef VIDERE_DEV_GLYPHMODEL_H
#define VIDERE_DEV_GLYPHMODEL_H

#include <string>
#include <vector>
#include <array>
#include <opencv2/core.hpp>
#include "Glyph.h"

#include <iostream>

namespace CudaImageProcLibsTrackHeadNS
{

    /**
     * Loads a vector of glyphs from a file and provides access to the data
     */
    class GlyphModel
    {

    public:
        GlyphModel();

        /**
         * Loads a model file containing glyphs and 3D points
         *
         * @param filename
         * @return
         */
        bool load(std::string filename, int scale=1);

        /**
         * Returns the number of glyphs loaded
         *
         * @return
         */
        size_t num_glyphs(){ return glyphs_.size();}

        /**
         * Returns a glyph at a particular index
         *
         * @param n
         * @return
         */
        Glyph get_glyph_by_index(size_t n);

        std::vector<cv::Point3f>& get_model();

    private:

        bool process_line(std::string line);

        void parse_num_glyphs(std::string line);

        void parse_glyph(std::string line);

        void parse_glyph_header(std::string line);

        void parse_glyph_pattern(std::string line, size_t &idx);

        void parse_glyph_vector(std::string line, size_t &idx);

        void add_new_glyph_to_vector();

    private:
        size_t num_glyphs_ = 0;
        std::vector<Glyph> glyphs_;

        char state_ = 'i';
        char state_glyph_ = 'h';

        size_t current_index_ = 0;
        std::array<std::array<uint8_t, 3>, 3> current_pattern_;
        std::array<float, 3> current_tl_;
        std::array<float, 3> current_tr_;
        std::array<float, 3> current_br_;
        std::array<float, 3> current_bl_;

        std::vector<cv::Point3f> model_;

    };

}
#endif //VIDERE_DEV_GLYPHMODEL_H
