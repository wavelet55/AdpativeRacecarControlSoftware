/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Jan. 2018
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
 *
 *  Common Image Processing Libs Types and Definitions
  *******************************************************************/

#ifndef VIDERE_DEV_PIXELCOLORVALUE_H
#define VIDERE_DEV_PIXELCOLORVALUE_H

#include <boost/math/constants/constants.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>

namespace ImageProcLibsNS
{

    enum ImageColorFormat_e
    {
        IPCF_RGB,
        IPCF_HSV,
        IPCF_HSL,
        IPCF_HSI,
        IPCF_YCrCb,
    };

    struct PixelColorValue_t
    {
        ImageColorFormat_e colorFormat = ImageColorFormat_e::IPCF_RGB;
        u_char c0 = 0;
        u_char c1 = 0;
        u_char c2 = 0;

        PixelColorValue_t() {}

        PixelColorValue_t(const PixelColorValue_t &cv)
        {
            colorFormat = cv.colorFormat;
            c0 = cv.c0;
            c1 = cv.c1;
            c2 = cv.c2;
        }

        PixelColorValue_t(unsigned int colorHexVal)
        {
            setColorWithFormat(colorHexVal);
        }

        void Clear()
        {
            colorFormat = ImageColorFormat_e::IPCF_RGB;
            c0 = 0;
            c1 = 0;
            c2 = 0;
        }

        void setRGBColor(unsigned int colorHexVal);

        //The colorHexVal value is a 4-byte value where:
        // b0 --> red or c0 color
        // b1 --> green or c1 color
        // b2 --> blue or c2 color
        // b3 --> color format (int)ImageColorFormat_e
        void setColorWithFormat(unsigned int colorHexVal);

        void setHSxColor(double hue, double Sat, double VLI, ImageColorFormat_e cfmt)
        {
            colorFormat = cfmt;
            setHueDegrees(hue);
            setSaturationPercent(Sat);
            setVLIPercent(VLI);
        }

        void setHueDegrees(double hue)
        {
            hue = (240.0 / 360.0) * hue;
            int iHue = (int)(hue + 0.5);
            iHue = iHue < 0 ? 0 : iHue > 239 ? 239 : iHue;
            c0 = (u_char)iHue;
        }

        double getHueDegrees()
        {
            double hue = (360.0 / 240.0) * (double)c0;
            return hue;
        }

        void setSaturationPercent(double val)
        {
            val = (240.0 / 100.0) * val;
            int iVal = (int)(val + 0.5);
            iVal = iVal < 0 ? 0 : iVal > 240 ? 240 : iVal;
            c1 = (u_char)iVal;
        }

        double getSaturationPercent()
        {
            double val = (100.0 / 240.0) * (double)c1;
            return val;
        }

        void setVLIPercent(double val)
        {
            val = (240.0 / 100.0) * val;
            int iVal = (int)(val + 0.5);
            iVal = iVal < 0 ? 0 : iVal > 240 ? 240 : iVal;
            c2 = (u_char)iVal;
        }

        double getVLIPercent()
        {
            double val = (100.0 / 240.0) * (double)c2;
            return val;
        }



        unsigned int ToUInt();

        void setRGBColor(u_char red, u_char green, u_char blue)
        {
            colorFormat = ImageColorFormat_e::IPCF_RGB;
            c0 = red;
            c1 = green;
            c2 = blue;
        }

        bool operator==(const PixelColorValue_t &cv)
        {
            bool tst = colorFormat == cv.colorFormat;
            tst &= c0 == cv.c0;
            tst &= c1 == cv.c1;
            tst &= c2 == cv.c2;
            return tst;
        }

        bool operator<(const PixelColorValue_t &cv)
        {
            bool tst = colorFormat == cv.colorFormat;
            tst &= c0 < cv.c0;
            tst &= c1 < cv.c1;
            tst &= c2 < cv.c2;
            return tst;
        }

        bool operator<=(const PixelColorValue_t &cv)
        {
            bool tst = colorFormat == cv.colorFormat;
            tst &= c0 <= cv.c0;
            tst &= c1 <= cv.c1;
            tst &= c2 <= cv.c2;
            return tst;
        }

        bool operator>(const PixelColorValue_t &cv)
        {
            bool tst = colorFormat == cv.colorFormat;
            tst &= c0 > cv.c0;
            tst &= c1 > cv.c1;
            tst &= c2 > cv.c2;
            return tst;
        }

        bool operator>=(const PixelColorValue_t &cv)
        {
            bool tst = colorFormat == cv.colorFormat;
            tst &= c0 >= cv.c0;
            tst &= c1 >= cv.c1;
            tst &= c2 >= cv.c2;
            return tst;
        }

        bool isHSxFormat()
        {
            bool hsx = colorFormat == ImageColorFormat_e::IPCF_HSV;
            hsx |= colorFormat == ImageColorFormat_e::IPCF_HSL;
            hsx |= colorFormat == ImageColorFormat_e::IPCF_HSI;
            return hsx;
        }

        u_char minRGBVal();
        u_char maxRGBVal();

        //Must be in RGB or BGR Format
        //Hue is in the range: [0, 360.0)
        double computeHue();


        //Must be in RGB or BGR Format
        //Chroma is in the range: [0, 100.0]
        double computeChroma();

        //Must be in RGB or BGR Format
        //Value is in the range: [0, 100.0]
        double computeHSV_V();

        //Must be in RGB or BGR Format
        //Luminance is in the range: [0, 100.0]
        double computeHSL_L();

        //Must be in RGB or BGR Format
        //Intensity is in the range: [0, 100.0]
        double computeHSI_I();


        //Change RGB or BGR to YCrCb format.
        //If already in that format.. noting happens.
        void RGBToYCrCbFormat();

        //Change RGB or BGR to HSV format.
        //If already in that format.. noting happens.
        //All values are in the range of 0 to 240
        void RGBToHSVFormat();

        //Change RGB or BGR to HSV format.
        //If already in that format.. noting happens.
        //All values are in the range of 0 to 240
        void RGBToHSLFormat();

        //Change RGB or BGR to HSV format.
        //If already in that format.. noting happens.
        //All values are in the range of 0 to 240
        void RGBToHSIFormat();

    };

}

#endif //VIDERE_DEV_PIXELCOLORVALUE_H
