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

#include "PixelColorValue.h"

namespace ImageProcLibsNS
{

    void PixelColorValue_t::setRGBColor(unsigned int colorHexVal)
    {
        colorFormat = ImageColorFormat_e::IPCF_RGB;
        c0 = (u_char)((colorHexVal >> 16) & 0xFF);
        c1 = (u_char)((colorHexVal >> 8) & 0xFF);
        c2 = (u_char)(colorHexVal & 0xFF);
    }

    //The colorHexVal value is a 4-byte value where:
    // b0 --> red or c0 color
    // b1 --> green or c1 color
    // b2 --> blue or c2 color
    // b3 --> color format (int)ImageColorFormat_e
    void PixelColorValue_t::setColorWithFormat(unsigned int colorHexVal)
    {
        colorFormat = (ImageColorFormat_e)((colorHexVal >> 24) & 0x7F);;
        c0 = (u_char)((colorHexVal >> 16) & 0xFF);
        c1 = (u_char)((colorHexVal >> 8) & 0xFF);
        c2 = (u_char)(colorHexVal & 0xFF);
    }

    unsigned int PixelColorValue_t::ToUInt()
    {
        unsigned int intVal = ((unsigned int)colorFormat) & 0x7F;
        intVal = (intVal << 8) + c0;
        intVal = (intVal << 8) + c1;
        intVal = (intVal << 8) + c2;
        return intVal;
    }


    //Change RGB or BGR to YCrCb (YUV) format.
    //If already in that format.. noting happens.
    void PixelColorValue_t::RGBToYCrCbFormat()
    {
        double Y, Cr, Cb;
        if(colorFormat == ImageColorFormat_e::IPCF_RGB)
        {
            Y = 0.257 * c0 + 0.504 * c1 + 0.098 * c2 + 16;
            Cr = 0.439 * c0 - 0.368 * c1 + -0.071 * c2 + 128;    // Cr image
            Cb = -0.148 * c0 - 0.2910 * c1 + 0.439 * c2 + 128;    // Cb image

            colorFormat = ImageColorFormat_e::IPCF_YCrCb;
            c0 = (u_char)Y;
            c1 = (u_char)Cr;
            c2 = (u_char)Cb;
        }
    }

    u_char PixelColorValue_t::minRGBVal()
    {
        u_char val = c0;
        val = val < c1 ? val : c1;
        val = val < c2 ? val : c2;
        return val;
    }

    u_char PixelColorValue_t::maxRGBVal()
    {
        u_char val = c0;
        val = val > c1 ? val : c1;
        val = val > c2 ? val : c2;
        return val;
    }

    //Must be in RGB or BGR Format
    //Hue is in the range: [0, 360.0)
    double PixelColorValue_t::computeHue()
    {
        double hue = 0;
        int R, G, B;

        if(isHSxFormat()) //Assume HSV, HSL, or HSI format
        {
            return (360.0 / 240.0) * c0;
        }

        R = c0;
        G = c1;
        B = c2;

        int Min = minRGBVal();
        int Max = maxRGBVal();
        double C = double(Max - Min);
        if(C == 0)
        {
            hue = 0.0;        //there is no correct Hue for this... so set to zero
        }
        else if(R == Max)
        {
            hue = (double)(G - B) / C;
            hue = hue < 0 ? 6.0 + hue : hue;    //ensures positive angles.
        }
        else if(G == Max)
        {
            hue = ((double)(B - R) / C) + 2.0;
        }
        else
        {
            hue = ((double)(R - G) / C) + 4.0;
        }
        //The above leaves Hue in the range of [0, 6.0)
        hue = 60.0 * hue;
        return hue;
    }

    //Must be in RGB or BGR Format
    //Chroma is in the range: [0, 100.0]
    double PixelColorValue_t::computeChroma()
    {
        double chroma = 0;
        int c = maxRGBVal() - minRGBVal();
        chroma = (100.0 / 255.0) * (double)c;
        return chroma;
    }


    //Must be in RGB or BGR Format
    //Value is in the range: [0, 100.0]
    double PixelColorValue_t::computeHSV_V()
    {
        double V = (100.0 / 255.0) * (double)maxRGBVal();
        return V;
    }

    //Must be in RGB or BGR Format
    //Luminance is in the range: [0, 100.0]
    double PixelColorValue_t::computeHSL_L()
    {
        double L = 0.5 * (double)(maxRGBVal() - minRGBVal());
        return L;
    }


    //Must be in RGB or BGR Format
    //Intensity is in the range: [0, 100.0]
    double PixelColorValue_t::computeHSI_I()
    {
        double I = (1.0 / 3.0) * (double)((int)c0 + (int)c1 + (int)c2);
        return I;
    }


    //Change RGB or BGR to HSV format.
    //If already in that format.. noting happens.
    //All values are in the range of 0 to 240
    void PixelColorValue_t::RGBToHSVFormat()
    {
        double H, V, C;

        if(colorFormat == ImageColorFormat_e::IPCF_RGB )
        {
            H = computeHue();
            C = computeChroma();
            V = computeHSV_V();

            c0 = (uint8_t)((240.0 / 360.0) * H);
            c1 = 0;
            if(V > 0)
            {
                c1 = (uint8_t)(240.0 * C / V);
            }
            c2 = (uint8_t)(2.40 * V);

            colorFormat = ImageColorFormat_e::IPCF_HSV;
        }
    }

    //Change RGB or BGR to HSV format.
    //If already in that format.. noting happens.
    //All values are in the range of 0 to 240
    void PixelColorValue_t::RGBToHSLFormat()
    {
        double H, V, C;

        if(colorFormat == ImageColorFormat_e::IPCF_RGB )
        {
            H = computeHue();
            C = computeChroma();
            V = computeHSL_L();

            c0 = (uint8_t)((240.0 / 360.0) * H);
            c1 = 0;
            if(V > 0 && V < 100.0)
            {

                c1 = (uint8_t)(240.0 * C / (100.0 - fabs(2.0 * V - 100.0)));
            }
            c2 = (uint8_t)(2.40 * V);

            colorFormat = ImageColorFormat_e::IPCF_HSL;
        }
    }

    //Change RGB or BGR to HSV format.
    //If already in that format.. noting happens.
    //All values are in the range of 0 to 240
    void PixelColorValue_t::RGBToHSIFormat()
    {
        double H, V, C;

        if(colorFormat == ImageColorFormat_e::IPCF_RGB )
        {
            H = computeHue();
            C = computeChroma();
            V = computeHSI_I();
            double m = (100.0 / 255.0) * (double)minRGBVal();

            c0 = (uint8_t)((240.0 / 360.0) * H);
            c1 = 0;
            if(V > 0)
            {
                c1 = (uint8_t)(240.0 * m / V);
            }
            c2 = (uint8_t)(2.40 * V);

            colorFormat = ImageColorFormat_e::IPCF_HSI;
        }
    }

}

