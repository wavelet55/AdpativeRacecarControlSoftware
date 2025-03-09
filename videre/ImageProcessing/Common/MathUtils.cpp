/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athrs: Randy Direen PhD, Harry Direen PhD
 * Date: July 2018
  *******************************************************************/


#include "MathUtils.h"

namespace MathLibsNS
{

    /**
    * @brief Reorder 2D points of a quadrilateral clockwise
    *
    * If you pass a vector of points to this function that define a
    * quadrilateral, the function will reorder them so that the top left point
    * is the first point, the top right is the second, the bottom right is
    * the third, and the bottom left is the fourth.
    *
    * @param points the unordered points to be put in order
    */
    void order_4_points(std::vector<cv::Point2f> &points)
    {
        //turn these into arrays
        std::array<cv::Point2f, 4> tmp;
        std::copy(points.begin(), points.end(), tmp.begin());

        std::array<float, 4> s;
        std::array<float, 4> d;

        for(int i = 0; i < 4; ++i)
        {
            s[i] = points[i].y + points[i].x;
            d[i] = points[i].y - points[i].x;
        }

        std::array<float, 4>::iterator it_r = std::min_element(s.begin(), s.end());
        points[0] = tmp[std::distance(s.begin(), it_r)];

        it_r = std::max_element(s.begin(), s.end());
        points[2] = tmp[std::distance(s.begin(), it_r)];

        it_r = std::min_element(d.begin(), d.end());
        points[1] = tmp[std::distance(d.begin(), it_r)];

        it_r = std::max_element(d.begin(), d.end());
        points[3] = tmp[std::distance(d.begin(), it_r)];
    }


    /**
    * @brief Max with and height of a parallelagram
    *
    * @param points
    *
    * @return a Point2f where x is width and y is height
    */
    cv::Point2f max_w_and_h(std::vector<cv::Point2f> points)
    {

        cv::Point2f tl = points[0];
        cv::Point2f tr = points[1];
        cv::Point2f br = points[2];
        cv::Point2f bl = points[3];

        cv::Point2f tdiff = tr - tl;
        cv::Point2f bdiff = br - bl;
        cv::Point2f ldiff = tl - bl;
        cv::Point2f rdiff = tr - br;

        float max_width;
        float m1 = sqrt(tdiff.x * tdiff.x + tdiff.y * tdiff.y);
        float m2 = sqrt(bdiff.x * bdiff.x + bdiff.y * bdiff.y);
        if(m1 > m2)
            max_width = m1;
        else
            max_width = m2;

        float max_height;
        m1 = sqrt(ldiff.x * ldiff.x + ldiff.y * ldiff.y);
        m2 = sqrt(rdiff.x * rdiff.x + rdiff.y * rdiff.y);
        if(m1 > m2)
            max_height = m1;
        else
            max_height = m2;

        return cv::Point(max_width, max_height);
    }

}
