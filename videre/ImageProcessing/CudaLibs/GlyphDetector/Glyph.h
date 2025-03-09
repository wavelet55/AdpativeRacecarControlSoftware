/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athrs: Randy Direen PhD, Harry Direen PhD
 * Date: August 2018
  *******************************************************************/

#ifndef VIDERE_DEV_GLYPH_H
#define VIDERE_DEV_GLYPH_H

#include <array>
#include <vector>
#include <opencv2/core.hpp>

namespace CudaImageProcLibsTrackHeadNS
{

#define PSIZE 25

    /**
     * Holds glyph pattern and 3D points; also provides methods for operating on glyphs
     *
     * Remember that a glyph has 0s around the border.
     */
    class Glyph {
    public:

        /**
         * Sets glyph pattern to zeros
         */
        Glyph();

        /**
         * Set the pattern and provides a bogus set of 3D locations
         *
         * Example:
         *  Glyph g()
         *  Glyph g1({0, 0, 0, 0, 0,
         *            0, 0, 0, 1, 0,
         *            0, 1, 0, 1, 0,
         *            0, 0, 0, 1, 0,
         *            0, 0, 0, 0, 0});
         *
         * @param pattern 25 element array of 1s and 0s
         */
        Glyph(std::array<uint8_t, PSIZE> pattern);

        Glyph(std::array<uint8_t, PSIZE> pattern,
              cv::Point3f top_left, cv::Point3f top_right,
              cv::Point3f bottom_left, cv::Point3f bottom_right);

        /**
         * Copy constructor
         *
         * @param g
         */
        Glyph(const Glyph& g);

        std::array<uint8_t, PSIZE> & pattern() {return pattern_;}

        cv::Point3f tl(){ return top_left_;}
        cv::Point3f tr(){ return top_right_;}
        cv::Point3f br(){ return bottom_right_;}
        cv::Point3f bl(){ return bottom_left_;}

        /**
         * XOR of the individual locations within the glyph
         *
         * @param g
         * @return
         */
        Glyph operator^(const Glyph& g);

        /**
         * Same thing as XOR, interpreted as an extension field where each element is the Galois Field GF(2)
         *
         * @param g
         * @return
         */
        Glyph operator+(const Glyph& g);

        /**
         * Magnitude is just the sum of the individual bits, or L1 norm
         *
         * @param g
         * @return
         */
        uint8_t friend mag(const Glyph& g);


        /**
         * Dot product, if same dot = 9 if orthogonal dot = 0
         *
         * @param g
         * @return
         */
        uint8_t friend dot(const Glyph& g1, const Glyph& g2);

        /**
         * Dot product over full set, dot = 9 if same and dot = 0 if orthogonal
         *
         * @param g
         * @return
         */
        uint8_t friend dotf(const Glyph& g1, const Glyph& g2);

        /**
         * Rotate the glyph pattern by 90 deg (this does nothing to the 3D points)
         *
         * @param g
         * @return
         */
        Glyph friend rotate_90deg_cw(const Glyph& g);

        /**
         * Rotate the glyph pattern by a multiple of 90 deg
         *
         * Examples:
         *  rotate_n_times(g, 1) //rotates by 90
         *  rotate_n_times(g, 2) //rotates by 180
         *  rotate_n_times(g, 3) //rotates by 270
         *
         * @param g
         * @param n
         * @return
         */
        Glyph friend rotate_n_times(const Glyph& g, size_t n);


    private:

        std::array<uint8_t, PSIZE> pattern_;
        cv::Point3f top_left_;
        cv::Point3f top_right_;
        cv::Point3f bottom_right_;
        cv::Point3f bottom_left_;

    };

    }

#endif //VIDERE_DEV_GLYPH_H
