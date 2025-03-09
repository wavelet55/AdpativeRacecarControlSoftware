/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athrs: Randy Direen PhD, Harry Direen PhD
 * Date: July 2018
  *******************************************************************/

#ifndef VIDERE_DEV_MATHUTILS_H
#define VIDERE_DEV_MATHUTILS_H

#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>
#include <armadillo>


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
    void order_4_points(std::vector<cv::Point2f> &points);


    /**
    * @brief Max with and height of a parallelagram
    *
    * @param points
    *
    * @return a Point2f where x is width and y is height
    */
    cv::Point2f max_w_and_h(std::vector<cv::Point2f> points);


    class MathUtils
    {

    };

}

#endif //VIDERE_DEV_MATHUTILS_H
