
#include "Glyph.h"


namespace CudaImageProcLibsTrackHeadNS
{

    Glyph::Glyph()
    {
        pattern_ = {0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0};

        top_left_ = cv::Point3f(-1, 1, 0);
        top_right_ = cv::Point3f(1, 1, 0);
        bottom_right_ = cv::Point3f(1, -1, 0);
        bottom_left_ = cv::Point3f(-1, -1, 0);
    }

    Glyph::Glyph(std::array<uint8_t, PSIZE> pattern)
    {
        pattern_ = pattern;
        top_left_ = cv::Point3f(-1, 1, 0);
        top_right_ = cv::Point3f(1, 1, 0);
        bottom_right_ = cv::Point3f(1, -1, 0);
        bottom_left_ = cv::Point3f(-1, -1, 0);
    }

    Glyph::Glyph(std::array<uint8_t, PSIZE> pattern, cv::Point3f top_left, cv::Point3f top_right, cv::Point3f bottom_left,
                 cv::Point3f bottom_right)
    {
        pattern_ = pattern;
        top_left_ = top_left;
        top_right_ = top_right;
        bottom_left_ = bottom_left;
        bottom_right_ = bottom_right;
    }

    Glyph::Glyph(const Glyph &g)
    {
        pattern_ = g.pattern_;
        top_left_ = g.top_left_;
        top_right_ = g.top_right_;
        bottom_left_ = g.bottom_left_;
        bottom_right_ = g.bottom_right_;
    }

    Glyph Glyph::operator^(const Glyph &g)
    {
        Glyph r;

        for(size_t i = 0; i < pattern_.size(); ++i)
        {
            r.pattern_[i] = this->pattern_[i]^g.pattern_[i];
        }

        return r;
    }

    Glyph Glyph::operator+(const Glyph &g)
    {
        Glyph r;

        r.pattern_[6] = this->pattern_[6]^g.pattern_[6];
        r.pattern_[7] = this->pattern_[7]^g.pattern_[7];
        r.pattern_[8] = this->pattern_[8]^g.pattern_[8];

        r.pattern_[11] = this->pattern_[11]^g.pattern_[11];
        r.pattern_[12] = this->pattern_[12]^g.pattern_[12];
        r.pattern_[13] = this->pattern_[13]^g.pattern_[13];

        r.pattern_[16] = this->pattern_[16]^g.pattern_[16];
        r.pattern_[17] = this->pattern_[17]^g.pattern_[17];
        r.pattern_[18] = this->pattern_[18]^g.pattern_[18];

        return r;
    }

    uint8_t dot(const Glyph &g1, const Glyph &g2)
    {
        uint8_t r = 0;
        r += 1 - g1.pattern_[6]^g2.pattern_[6];
        r += 1 - g1.pattern_[7]^g2.pattern_[7];
        r += 1 - g1.pattern_[8]^g2.pattern_[8];

        r += 1 - g1.pattern_[11]^g2.pattern_[11];
        r += 1 - g1.pattern_[12]^g2.pattern_[12];
        r += 1 - g1.pattern_[13]^g2.pattern_[13];

        r += 1 - g1.pattern_[16]^g2.pattern_[16];
        r += 1 - g1.pattern_[17]^g2.pattern_[17];
        r += 1 - g1.pattern_[18]^g2.pattern_[18];
        return r;
    }

    uint8_t dotf(const Glyph &g1, const Glyph &g2)
    {
        uint8_t r = 0;
        for(size_t i = 0; i < g1.pattern_.size(); ++i)
        {
            r += 1 - g1.pattern_[i]^g2.pattern_[i];
        }

        return r;
    }

    uint8_t mag(const Glyph& g)
    {
        uint8_t r = 0;

        r += g.pattern_[6];
        r += g.pattern_[7];
        r += g.pattern_[8];

        r += g.pattern_[11];
        r += g.pattern_[12];
        r += g.pattern_[13];

        r += g.pattern_[16];
        r += g.pattern_[17];
        r += g.pattern_[18];

        return r;

    }

    Glyph rotate_90deg_cw(const Glyph &g)
    {
        Glyph r;

        r.pattern_[6] = g.pattern_[16];
        r.pattern_[7] = g.pattern_[11];
        r.pattern_[8] = g.pattern_[6];

        r.pattern_[11] = g.pattern_[17];
        r.pattern_[12] = g.pattern_[12];
        r.pattern_[13] = g.pattern_[7];

        r.pattern_[16] = g.pattern_[18];
        r.pattern_[17] = g.pattern_[13];
        r.pattern_[18] = g.pattern_[8];
        return r;

    }

    Glyph rotate_n_times(const Glyph &g, size_t n)
    {
        Glyph r(g);

        for(size_t i = 0; i < n; ++i)
            r = rotate_90deg_cw(r);

        return r;
    }



}
