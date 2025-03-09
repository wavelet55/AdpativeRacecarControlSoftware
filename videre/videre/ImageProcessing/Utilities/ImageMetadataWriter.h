/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Nov. 2017
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
  *******************************************************************/

#ifndef VIDERE_DEV_IMAGEMETADATAWRITER_H
#define VIDERE_DEV_IMAGEMETADATAWRITER_H

#include "global_defines.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "../../Utils/logger.h"
#include <string>

namespace VidereImageprocessing
{

    enum ImageTextLocation_e
    {
        ITL_UpperLeftCorner,
        ITL_UpperRightCorner,
        ITL_LowerLeftCorner,
        ITL_LowerRightCorner
    };

    enum ImageTextColor_e
    {
        ITC_Black,
        ITC_White,
    };

    ///ImageMetadataWriter
    ///Used to write text such as image number and timestamp
    ///on an image.
    class ImageMetadataWriter
    {

    public:
        int TextFont = (int)cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;

        double TextScaleFactor = 0.75;

        cv::Scalar TextColor;

        cv::Scalar BackgroundColor;

        double VerticalOffsetFromEdgeScaleFactor = 0.05;

        double HorizontalOffsetFromEdgeScaleFactor = 0.05;

        int TextThickness = 1;

        int LineType = (int)cv::LineTypes::LINE_8;

        cv::Size NominalTextSize;

        int NominalTextBaseLine = 0;

        cv::Point TextRegionUpperLeft;

        cv::Point TextRegionLowerRight;

        int TextYOffsetFromTextRegionBottom = 0;

        bool SolidTextBackGround = true;


    public:
        ImageMetadataWriter()
        {
            SetNominalTextSize();
            SetTextScreenLocation(true, true, true, 640, 480);
        }

        //void SetFont();

        cv::Scalar GetColorVal(ImageTextColor_e color)
        {
            cv::Scalar colorVal = (0, 0, 0);  //Black
            if(color == ITC_White) {
                colorVal[0] = 255;
                colorVal[1] = 255;
                colorVal[2] = 255;
            } else {
                colorVal[0] = 0;
                colorVal[1] = 0;
                colorVal[2] = 0;
            }
            return colorVal;
        }

        void SetFontColor(ImageTextColor_e color)
        {
            TextColor = GetColorVal(color);
        }

        void SetBackgroundColor(ImageTextColor_e color)
        {
            BackgroundColor = GetColorVal(color);
        }

        void SetTextActualSize(const std::string &textStr);

        void SetNominalTextSize()
        {
            std::string exTextStr = "ImageNo: 001";
            SetTextActualSize(exTextStr);
        }

        cv::Point ComputeStartPoint(ImageTextLocation_e loc, int imgWidth, int imgHeight, int textLen);

        void SetTextScreenLocation(bool topScreen, bool textColorWhite, bool solidBackground,
                                        int imageWidth, int imageHeight);

        void WriteInt(cv::Mat &image, const std::string *msg, int val, cv::Point startPt);

        void WriteDouble(cv::Mat &image, const std::string *msg, double val, int precision, cv::Point startPt);

        void WriteImageNoAndTimeStamp(cv::Mat &image, unsigned int imgNo, double ts );

    };

}
#endif //VIDERE_DEV_IMAGEMETADATAWRITER_H
