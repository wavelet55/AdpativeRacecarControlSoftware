
#include "GlyphModel.h"
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>

namespace CudaImageProcLibsTrackHeadNS
{

    GlyphModel::GlyphModel()
    {

    }

    bool GlyphModel::load(std::string filename, int scale)
    {
        bool r = true;
        try
        {
            boost::iostreams::stream<boost::iostreams::file_source> file(filename.c_str());
            std::string line;
            while(std::getline(file, line))
            {
                if((line[0] != '#') && (line[0] != ' ') && (line[0] != '\r') && !line.empty())
                {
                    bool rr = process_line(line);
                    if(!rr)
                        r = false;
                }

            }

            if(state_ == 'i')
                return false;

            for(int i = 0; i < glyphs_.size(); ++i)
            {
                model_.push_back(scale * glyphs_[i].tl());
                model_.push_back(scale * glyphs_[i].tr());
                model_.push_back(scale * glyphs_[i].br());
                model_.push_back(scale * glyphs_[i].bl());
            }
        }
        catch (std::exception &e)
        {
            r = false;
        }

        return r;
    }

    Glyph GlyphModel::get_glyph_by_index(size_t n)
    {
        return glyphs_[n];
    }

    std::vector<cv::Point3f> &GlyphModel::get_model()
    {
        return model_;
    }

    bool GlyphModel::process_line(std::string line)
    {
        switch(state_)
        {
            case 'i':
                parse_num_glyphs(line);
                break;
            case 'g':
                parse_glyph(line);
                break;
        }
        return true;
    }

    void GlyphModel::parse_num_glyphs(std::string line)
    {

        std::vector<std::string> strs;
        boost::split(strs,line,boost::is_any_of("="));

        if(strs.size() >= 2)
        {
            strs[0].erase(std::remove_if(strs[0].begin(), strs[0].end(), isspace), strs[0].end());
            strs[1].erase(std::remove_if(strs[1].begin(), strs[1].end(), isspace), strs[1].end());
            state_ = 'g';
            if(boost::iequals(strs[0], "NumGlyphs"))
            {
                num_glyphs_ = boost::lexical_cast<size_t>(strs[1]);
            }
        }
    }

    void GlyphModel::parse_glyph(std::string line)
    {

        static size_t gidx = 0;
        static size_t vidx = 0;

        switch(state_glyph_)
        {
            case 'h': //header of glyph
                parse_glyph_header(line);
                break;

            case 'p': //parse the pattern
                parse_glyph_pattern(line, gidx);
                break;

            case 'v': //parse the vector
                parse_glyph_vector(line, vidx);
                break;

            case 'c': //add new glyph to vector
                break;

        }

    }

    void GlyphModel::parse_glyph_header(std::string line)
    {
        std::vector<std::string> strs;
        boost::split(strs,line,boost::is_any_of(":"));

        if(strs.size() == 2)
        {
            strs[0].erase(std::remove_if(strs[0].begin(), strs[0].end(), isspace), strs[0].end());
            strs[1].erase(std::remove_if(strs[1].begin(), strs[1].end(), isspace), strs[1].end());
            state_glyph_ = 'p';
            if(boost::iequals(strs[0], "Glyph"))
            {
                current_index_ = boost::lexical_cast<size_t>(strs[1]);
            }
        }
    }

    void GlyphModel::parse_glyph_pattern(std::string line, size_t &idx)
    {
        std::vector<std::string> strs;
        boost::split(strs,line,boost::is_any_of(" "));

        strs[0].erase(std::remove_if(strs[0].begin(), strs[0].end(), isspace), strs[0].end());
        strs[1].erase(std::remove_if(strs[1].begin(), strs[1].end(), isspace), strs[1].end());
        strs[2].erase(std::remove_if(strs[2].begin(), strs[2].end(), isspace), strs[2].end());

        if(strs.size() == 3){
            current_pattern_[idx][0] = static_cast<uint8_t>(boost::lexical_cast<int>(strs[0]));
            current_pattern_[idx][1] = static_cast<uint8_t>(boost::lexical_cast<int>(strs[1]));
            current_pattern_[idx][2] = static_cast<uint8_t>(boost::lexical_cast<int>(strs[2]));
        }


        ++idx;
        if(idx == 3)
        {
            idx = 0;
            state_glyph_ = 'v';
        }

    }

    void GlyphModel::parse_glyph_vector(std::string line, size_t &idx)
    {

        std::vector<std::string> strs;
        boost::split(strs,line,boost::is_any_of("="));

        if(strs.size() == 2)
        {
            strs[0].erase(std::remove_if(strs[0].begin(), strs[0].end(), isspace), strs[0].end());

            std::vector<std::string> strs2;
            boost::split(strs2, strs[1], boost::is_any_of(","));
            if(strs2.size() == 3){
                strs2[0].erase(std::remove_if(strs2[0].begin(), strs2[0].end(), isspace), strs2[0].end());
                strs2[1].erase(std::remove_if(strs2[1].begin(), strs2[1].end(), isspace), strs2[1].end());
                strs2[2].erase(std::remove_if(strs2[2].begin(), strs2[2].end(), isspace), strs2[2].end());
                if(boost::iequals(strs[0], "tl"))
                {
                    current_tl_[0] = boost::lexical_cast<float>(strs2[0]);
                    current_tl_[1] = boost::lexical_cast<float>(strs2[1]);
                    current_tl_[2] = boost::lexical_cast<float>(strs2[2]);
                }
                if(boost::iequals(strs[0], "tr"))
                {
                    current_tr_[0] = boost::lexical_cast<float>(strs2[0]);
                    current_tr_[1] = boost::lexical_cast<float>(strs2[1]);
                    current_tr_[2] = boost::lexical_cast<float>(strs2[2]);
                }
                if(boost::iequals(strs[0], "br"))
                {
                    current_br_[0] = boost::lexical_cast<float>(strs2[0]);
                    current_br_[1] = boost::lexical_cast<float>(strs2[1]);
                    current_br_[2] = boost::lexical_cast<float>(strs2[2]);
                }
                if(boost::iequals(strs[0], "bl"))
                {
                    current_bl_[0] = boost::lexical_cast<float>(strs2[0]);
                    current_bl_[1] = boost::lexical_cast<float>(strs2[1]);
                    current_bl_[2] = boost::lexical_cast<float>(strs2[2]);
                }
            }
        }

        ++idx;
        if(idx == 4)
        {
            add_new_glyph_to_vector();
            idx = 0;
            state_glyph_ = 'h';
        }
    }

    void GlyphModel::add_new_glyph_to_vector()
    {

        std::array<uint8_t, PSIZE> pattern;
        std::fill(pattern.begin(), pattern.end(), 0);

        pattern[6] = current_pattern_[0][0];
        pattern[7] = current_pattern_[0][1];
        pattern[8] = current_pattern_[0][2];
        std::fill(current_pattern_[0].begin(), current_pattern_[0].end(), 0);

        pattern[11] = current_pattern_[1][0];
        pattern[12] = current_pattern_[1][1];
        pattern[13] = current_pattern_[1][2];
        std::fill(current_pattern_[1].begin(), current_pattern_[1].end(), 0);

        pattern[16] = current_pattern_[2][0];
        pattern[17] = current_pattern_[2][1];
        pattern[18] = current_pattern_[2][2];
        std::fill(current_pattern_[2].begin(), current_pattern_[2].end(), 0);

        cv::Point3f tl(current_tl_[0], current_tl_[1], current_tl_[2]);
        std::fill(current_tl_.begin(), current_tl_.end(), 0);

        cv::Point3f tr(current_tr_[0], current_tr_[1], current_tr_[2]);
        std::fill(current_tr_.begin(), current_tr_.end(), 0);

        cv::Point3f br(current_br_[0], current_br_[1], current_br_[2]);
        std::fill(current_br_.begin(), current_br_.end(), 0);

        cv::Point3f bl(current_bl_[0], current_bl_[1], current_bl_[2]);
        std::fill(current_bl_.begin(), current_bl_.end(), 0);

        Glyph g(pattern, tl, tr,
                         bl, br);

        glyphs_.push_back(g);
    }



}
