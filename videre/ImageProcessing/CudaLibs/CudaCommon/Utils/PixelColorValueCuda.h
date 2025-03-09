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

#ifndef VIDERE_DEV_PIXELCOLORVALUECUDA_H
#define VIDERE_DEV_PIXELCOLORVALUECUDA_H

#include <cuda_runtime.h>

#pragma warning(disable:4270)

namespace CudaImageProcLibsNS
{

    enum ImageColorFormatCuda_e
    {
        IPCF_RGB,
        IPCF_HSV,
        IPCF_HSL,
        IPCF_HSI,
        IPCF_YCrCb,
    };

    class PixelColorValueCuda_t
    {
    public:

        ImageColorFormatCuda_e colorFormat;
        unsigned char c0;
        unsigned char c1;
        unsigned char c2;

        __device__
        inline PixelColorValueCuda_t()
        {
            colorFormat = ImageColorFormatCuda_e::IPCF_RGB;
            c0 = 0;
            c1 = 0;
            c2 = 0;
        }

        __device__
        inline PixelColorValueCuda_t(const PixelColorValueCuda_t &cv)
        {
            colorFormat = cv.colorFormat;
            c0 = cv.c0;
            c1 = cv.c1;
            c2 = cv.c2;
        }

        __device__
        inline PixelColorValueCuda_t(unsigned int colorHexVal)
        {
            colorFormat = (ImageColorFormatCuda_e)((colorHexVal >> 24) & 0x7F);
            c0 = (unsigned char)((colorHexVal >> 16) & 0xFF);
            c1 = (unsigned char)((colorHexVal >> 8) & 0xFF);
            c2 = (unsigned char)(colorHexVal & 0xFF);
        }

        __device__
        inline void Clear()
        {
            colorFormat = ImageColorFormatCuda_e::IPCF_RGB;
            c0 = 0;
            c1 = 0;
            c2 = 0;
        }

        __device__
        inline void setRGBColor(unsigned int colorHexVal)
        {
            colorFormat = ImageColorFormatCuda_e::IPCF_RGB;
            c0 = (unsigned char)((colorHexVal >> 16) & 0xFF);
            c1 = (unsigned char)((colorHexVal >> 8) & 0xFF);
            c2 = (unsigned char)(colorHexVal & 0xFF);
        }

        __device__
        inline void setRGBColor(unsigned char red, unsigned char green, unsigned char blue)
        {
            colorFormat = ImageColorFormatCuda_e::IPCF_RGB;
            c0 = red;
            c1 = green;
            c2 = blue;
        }

        inline bool isHSxFormat()
        {
            bool hsx = colorFormat == ImageColorFormatCuda_e::IPCF_HSV;
            hsx |= colorFormat == ImageColorFormatCuda_e::IPCF_HSL;
            hsx |= colorFormat == ImageColorFormatCuda_e::IPCF_HSI;
            return hsx;
        }

        //The colorHexVal value is a 4-byte value where:
        // b0 --> red or c0 color
        // b1 --> green or c1 color
        // b2 --> blue or c2 color
        // b3 --> color format (int)ImageColorFormatCuda_e
        __device__
        inline void setColorWithFormat(unsigned int colorHexVal)
        {
            colorFormat = (ImageColorFormatCuda_e)((colorHexVal >> 24) & 0x7F);
            c0 = (unsigned char)((colorHexVal >> 16) & 0xFF);
            c1 = (unsigned char)((colorHexVal >> 8) & 0xFF);
            c2 = (unsigned char)(colorHexVal & 0xFF);
        }

        //convert to a Unsigned 32-bit Int
        __device__
        inline unsigned int ToUInt()
        {
            unsigned int intVal = ((unsigned int)colorFormat) & 0x7F;
            intVal = (intVal << 8) + c0;
            intVal = (intVal << 8) + c1;
            intVal = (intVal << 8) + c2;
            return intVal;
        }

        __device__
        inline bool operator==(const PixelColorValueCuda_t &cv)
        {
            bool tst = colorFormat == cv.colorFormat;
            tst &= c0 == cv.c0;
            tst &= c1 == cv.c1;
            tst &= c2 == cv.c2;
            return tst;
        }

        __device__
        inline bool operator<(const PixelColorValueCuda_t &cv)
        {
            bool tst = colorFormat == cv.colorFormat;
            tst &= c0 < cv.c0;
            tst &= c1 < cv.c1;
            tst &= c2 < cv.c2;
            return tst;
        }

        __device__
        inline bool operator<=(const PixelColorValueCuda_t &cv)
        {
            bool tst = colorFormat == cv.colorFormat;
            tst &= c0 <= cv.c0;
            tst &= c1 <= cv.c1;
            tst &= c2 <= cv.c2;
            return tst;
        }

        __device__
        inline bool operator>(const PixelColorValueCuda_t &cv)
        {
            bool tst = colorFormat == cv.colorFormat;
            tst &= c0 > cv.c0;
            tst &= c1 > cv.c1;
            tst &= c2 > cv.c2;
            return tst;
        }

        __device__
        inline bool operator>=(const PixelColorValueCuda_t &cv)
        {
            bool tst = colorFormat == cv.colorFormat;
            tst &= c0 >= cv.c0;
            tst &= c1 >= cv.c1;
            tst &= c2 >= cv.c2;
            return tst;
        }


        //Change RGB or BGR to YCrCb (YUV) format.
        //If already in that format.. noting happens.
        __device__
        inline void RGBToYCrCbFormat()
        {
            float Y, Cr, Cb;
            if(colorFormat == ImageColorFormatCuda_e::IPCF_RGB)
            {
                Y = 0.257 * c0 + 0.504 * c1 + 0.098 * c2 + 16;
                Cr = 0.439 * c0 - 0.368 * c1 + -0.071 * c2 + 128;    // Cr image
                Cb = -0.148 * c0 - 0.2910 * c1 + 0.439 * c2 + 128;    // Cb image

                colorFormat = ImageColorFormatCuda_e::IPCF_YCrCb;
                c0 = (u_char)Y;
                c1 = (u_char)Cr;
                c2 = (u_char)Cb;
            }
        }

        __device__
        inline u_char minRGBVal()
        {
            u_char val = c0;
            val = val < c1 ? val : c1;
            val = val < c2 ? val : c2;
            return val;
        }

        __device__
        inline u_char maxRGBVal()
        {
            u_char val = c0;
            val = val > c1 ? val : c1;
            val = val > c2 ? val : c2;
            return val;
        }

        //Must be in RGB or BGR Format
        //Hue is in the range: [0, 360.0)
        __device__
        inline float computeHue()
        {
            float hue = 0;
            int R, G, B;

            R = c0;
            G = c1;
            B = c2;
            int Min = minRGBVal();
            int Max = maxRGBVal();
            float C = float(Max - Min);
            if(C == 0)
            {
                hue = 0.0;        //there is no correct Hue for this... so set to zero
            }
            else if(R == Max)
            {
                hue = (float)(G - B) / C;
                hue = hue < 0 ? 6.0 + hue : hue;    //ensures positive angles.
            }
            else if(G == Max)
            {
                hue = ((float)(B - R) / C) + 2.0;
            }
            else
            {
                hue = ((float)(R - G) / C) + 4.0;
            }
            //The above leaves Hue in the range of [0, 6.0)
            hue = 60.0 * hue;
            return hue;
        }

        //Must be in RGB or BGR Format
        //Chroma is in the range: [0, 100.0]
        __device__
        inline float computeChroma()
        {
            float chroma = 0;
            int c = maxRGBVal() - minRGBVal();
            chroma = (100.0 / 255.0) * (float)c;
            return chroma;
        }


        //Must be in RGB or BGR Format
        //Value is in the range: [0, 100.0]
        __device__
        inline float computeHSV_V()
        {
            float V = (100.0 / 255.0) * (float)maxRGBVal();
            return V;
        }

        //Must be in RGB or BGR Format
        //Luminance is in the range: [0, 100.0]
        __device__
        inline float computeHSL_L()
        {
            float L = 0.5 * (float)(maxRGBVal() - minRGBVal());
            return L;
        }

        //Must be in RGB or BGR Format
        //Intensity is in the range: [0, 100.0]
        __device__
        inline float computeHSI_I()
        {
            float I = (1.0 / 3.0) * (float)((int)c0 + (int)c1 + (int)c2);
            return I;
        }

        //Change RGB or BGR to HSV format.
        //If already in that format.. noting happens.
        //All values are in the range of 0 to 240
        __device__
        inline void RGBToHSVFormat()
        {
            float H, V, C;

            if(colorFormat == ImageColorFormatCuda_e::IPCF_RGB)
            {
                H = computeHue();
                C = computeChroma();
                V = computeHSV_V();

                c0 = (u_char)((240.0 / 360.0) * H);
                c1 = 0;
                if(V > 0)
                {
                    c1 = (u_char)(240.0 * C / V);
                }
                c2 = (u_char)(2.40 * V);

                colorFormat = ImageColorFormatCuda_e::IPCF_HSV;
            }
        }

        //Change RGB or BGR to HSV format.
        //If already in that format.. noting happens.
        //All values are in the range of 0 to 240
        __device__
        inline void RGBToHSLFormat()
        {
            float H, V, C;

            if(colorFormat == ImageColorFormatCuda_e::IPCF_RGB)
            {
                H = computeHue();
                C = computeChroma();
                V = computeHSL_L();

                c0 = (u_char)((240.0 / 360.0) * H);
                c1 = 0;
                if(V > 0 && V < 100.0)
                {

                    c1 = (u_char)(240.0 * C / (100.0 - fabs(2.0 * V - 100.0)));
                }
                c2 = (u_char)(2.40 * V);

                colorFormat = ImageColorFormatCuda_e::IPCF_HSL;
            }
        }

        //Change RGB or BGR to HSV format.
        //If already in that format.. noting happens.
        //All values are in the range of 0 to 240
        __device__
        inline void RGBToHSIFormat()
        {
            float H, V, C;

            if(colorFormat == ImageColorFormatCuda_e::IPCF_RGB)
            {
                H = computeHue();
                C = computeChroma();
                V = computeHSI_I();
                float m = (100.0 / 255.0) * (float)minRGBVal();

                c0 = (u_char)((240.0 / 360.0) * H);
                c1 = 0;
                if(V > 0)
                {
                    c1 = (u_char)(240.0 * m / V);
                }
                c2 = (u_char)(2.40 * V);

                colorFormat = ImageColorFormatCuda_e::IPCF_HSI;
            }
        }



    };

}

#endif //VIDERE_DEV_PIXELCOLORVALUE_H
