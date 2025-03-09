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

#include "ImageMetadataWriter.h"
#include "opencv2/imgproc.hpp"
#include <iomanip> // setprecision
#include <sstream> // stringstream

using namespace std;
using namespace cv;

namespace VidereImageprocessing
{
    cv::Point ImageMetadataWriter::ComputeStartPoint(ImageTextLocation_e loc, int imgWidth, int imgHeight, int textLen)
    {
        cv::Point startPt;
        switch(loc)
        {
            case  ITL_UpperLeftCorner:
                startPt.x = (int)(HorizontalOffsetFromEdgeScaleFactor * imgWidth);
                startPt.y = (int)(VerticalOffsetFromEdgeScaleFactor * imgHeight);
                break;
            case  ITL_UpperRightCorner:
                //startPt.x = (int)(HorizontalOffsetFromEdgeScaleFactor * imgWidth);
                //startPt.x = imgWidth - startPt.x - textLen;
                //startPt.x = startPt.x < imgWidth / 2 ? imgWidth / 2 : startPt.x;
                startPt.x = imgWidth >> 1;
                startPt.y = (int)(VerticalOffsetFromEdgeScaleFactor * imgHeight);

                break;
            case  ITL_LowerLeftCorner:
                startPt.x = (int)(HorizontalOffsetFromEdgeScaleFactor * imgWidth);
                startPt.y = (int)(VerticalOffsetFromEdgeScaleFactor * imgHeight);
                startPt.y = imgHeight - startPt.y;
                break;
            case  ITL_LowerRightCorner:
                //startPt.x = (int)(HorizontalOffsetFromEdgeScaleFactor * imgWidth);
                //startPt.x = imgWidth - startPt.x - textLen;
                //startPt.x = startPt.x < imgWidth / 2 ? imgWidth / 2 : startPt.x;
                startPt.x = imgWidth >> 1;
                startPt.y = (int)(VerticalOffsetFromEdgeScaleFactor * imgHeight);
                startPt.y = imgHeight - startPt.y;
                break;
        }
        return startPt;
    }

    void ImageMetadataWriter::SetTextActualSize(const std::string &textStr)
    {
        NominalTextSize = cv::getTextSize(textStr, TextFont,
                                                TextScaleFactor, TextThickness, &NominalTextBaseLine);
    }

    void ImageMetadataWriter::WriteInt(cv::Mat &image, const std::string *msg, int val, cv::Point startPt)
    {
        stringstream textStr;
        if(msg != nullptr)
        {
            textStr << *msg << val;
        } else {
            textStr << val;
        }

        cv::putText(image, textStr.str().c_str(), startPt,
                    TextFont, TextScaleFactor, TextColor, TextThickness, LineType, false);
    }

    void ImageMetadataWriter::WriteDouble(cv::Mat &image, const std::string *msg,
                                          double val, int precision, cv::Point startPt)
    {
        precision = precision < 1 ? 1 : precision > 6 ? 6 : precision;
        stringstream textStr;
        if(msg != nullptr)
        {
            textStr << *msg  << std::fixed << std::setprecision(precision) << val;
        } else
        {
            textStr << std::fixed << std::setprecision(precision) << val;
        }

        cv::putText(image, textStr.str().c_str(), startPt,
                    TextFont, TextScaleFactor, TextColor, TextThickness, LineType, false);
    }


    void ImageMetadataWriter::SetTextScreenLocation(bool topScreen,
                                                    bool textColorWhite,
                                                    bool solidBackground,
                                                    int imageWidth, int imageHeight)
    {
        int bkgndStripWidth = 2 * NominalTextSize.height;
        TextYOffsetFromTextRegionBottom = NominalTextSize.height >> 1;  //1/4th
        SolidTextBackGround = solidBackground;

        if(topScreen)
        {
            TextRegionUpperLeft.x = 0;
            TextRegionUpperLeft.y = 0;
            TextRegionLowerRight.x = imageWidth - 1;
            TextRegionLowerRight.y = bkgndStripWidth;
        }
        else
        {
            TextRegionUpperLeft.x = 0;
            TextRegionUpperLeft.y = imageHeight - 1 - bkgndStripWidth;
            TextRegionLowerRight.x = imageWidth - 1;
            TextRegionLowerRight.y = imageHeight - 1;
        }

        if(textColorWhite)
        {
            SetFontColor(ITC_White);
            SetBackgroundColor(ITC_Black);

        }
        else
        {
            SetFontColor(ITC_Black);
            SetBackgroundColor(ITC_White);
        }
    }



    void ImageMetadataWriter::WriteImageNoAndTimeStamp(cv::Mat &image, unsigned int imgNo, double ts)
    {
        cv::Point textStartPt;

        if(SolidTextBackGround)
        {
            cv::rectangle(image, TextRegionUpperLeft, TextRegionLowerRight, BackgroundColor, cv::FILLED);
        }

        textStartPt.x = image.cols >> 4;
        textStartPt.y = TextRegionLowerRight.y - TextYOffsetFromTextRegionBottom;
        string msg = "Image No: ";
        WriteInt(image, &msg, imgNo, textStartPt);

        textStartPt.x = image.cols >> 1;
        textStartPt.y = TextRegionLowerRight.y - TextYOffsetFromTextRegionBottom;
        msg = "Time(sec): ";
        WriteDouble(image, &msg, ts, 3, textStartPt);
    }

}