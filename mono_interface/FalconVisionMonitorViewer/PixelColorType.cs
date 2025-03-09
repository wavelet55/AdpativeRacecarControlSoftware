using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FalconVisionMonitorViewer
{
    public enum ImageColorFormat_e
    {
        RGB,
        HSV,
        HSL,
        HSI,
        YCrCb,
    };

    public class PixelColorValue_t
    {

        public ImageColorFormat_e colorFormat = ImageColorFormat_e.RGB;
        //The raw color values are contained in a byte.
        //Red, Green, Blue values are in the range [0, 255];
        //Raw Hue is in the range of [0, 239] ... 240 would be == 360 degrees
        //Saturation, V, I, L are in the range [0, 240] where 240 == 100%.
        //Y, Cr, Cb values are in the range [0, 255]... but don't necessarily use the whole range.
        public byte c0 = 0;
        public byte c1 = 0;
        public byte c2 = 0;

        //The Getter/Setters assume the class is in the correct format for the value
        //being accessed... if not... invalid values may be set or obtained.  
        //The user beware.   

        public byte Red
        {
            get { return c0; }
            set { c0 = value; }
        }

        public byte Green
        {
            get { return c1; }
            set { c1 = value; }
        }

        public byte Blue
        {
            get { return c2; }
            set { c2 = value; }
        }


        public byte Y
        {
            get { return c0; }
            set { c0 = value; }
        }

        public byte Cr
        {
            get { return c1; }
            set { c1 = value; }
        }

        public byte Cb
        {
            get { return c2; }
            set { c2 = value; }
        }


        public double HueDegrees
        {
            get { return (360.0 / 240.0) * (double)c0; }
            set 
            {
                double tmp = (240.0 / 360.0) * value;
                tmp = Math.Round(tmp);
                tmp = tmp < 0 ? 0 : tmp > 239 ? 239 : tmp;
                c0 = (byte)tmp;
            }
        }

        public double SaturationPercent
        {
            get { return (100.0 / 240.0) * (double)c1; }
            set 
            {
                double tmp = (240.0 / 100.0) * value;
                tmp = Math.Round(tmp);
                tmp = tmp < 0 ? 0 : tmp > 240 ? 240 : tmp;
                c1 = (byte)tmp;
            }
        }

        //Lumanance, Value, or Intensity Percent
        public double VLIPercent
        {
            get { return (100.0 / 240.0) * (double)c2; }
            set 
            {
                double tmp = (240.0 / 100.0) * value;
                tmp = Math.Round(tmp);
                tmp = tmp < 0 ? 0 : tmp > 240 ? 240 : tmp;
                c2 = (byte)tmp;
            }
        }


        public PixelColorValue_t() 
        {
            Clear();
        }

        public PixelColorValue_t(PixelColorValue_t cv)
        {
            colorFormat = cv.colorFormat;
            c0 = cv.c0;
            c1 = cv.c1;
            c2 = cv.c2;
        }

        public PixelColorValue_t(UInt32 colorHexVal)
        {
            setColorWithFormat(colorHexVal);
        }

        void Clear()
        {
            colorFormat = ImageColorFormat_e.RGB;
            c0 = 0;
            c1 = 0;
            c2 = 0;
        }

        public void setRGBColor(UInt32 colorHexVal)
        {
            colorFormat = ImageColorFormat_e.RGB;
            c0 = (byte)((colorHexVal >> 16) & 0xFF);
            c1 = (byte)((colorHexVal >> 8) & 0xFF);
            c2 = (byte)(colorHexVal & 0xFF);
        }

        //The colorHexVal value is a 4-byte value where:
        // b0 --> red or c0 color
        // b1 --> green or c1 color
        // b2 --> blue or c2 color
        // b3 --> color format (int)ImageColorFormat_e
        void setColorWithFormat(UInt32 colorHexVal)
        {
            colorFormat = (ImageColorFormat_e)((colorHexVal >> 24) & 0x7F);
            c0 = (byte)((colorHexVal >> 16) & 0xFF);
            c1 = (byte)((colorHexVal >> 8) & 0xFF);
            c2 = (byte)(colorHexVal & 0xFF);
        }

        public void CopyFrom(PixelColorValue_t cv)
        {
            colorFormat = cv.colorFormat;
            c0 = cv.c0;
            c1 = cv.c1;
            c2 = cv.c2;
        }

        public UInt32 ToUInt()
        {
            UInt32 intVal = ((UInt32)colorFormat) & 0x7F;
            intVal = (intVal << 8) + c0;
            intVal = (intVal << 8) + c1;
            intVal = (intVal << 8) + c2;
            return intVal;
        }

        public void setRGBColor(byte red, byte green, byte blue)
        {
            colorFormat = ImageColorFormat_e.RGB;
            c0 = red;
            c1 = green;
            c2 = blue;
        }


        public bool isHSxFormat()
        {
            bool hsx = colorFormat == ImageColorFormat_e.HSV;
            hsx |= colorFormat == ImageColorFormat_e.HSL;
            hsx |= colorFormat == ImageColorFormat_e.HSI;
            return hsx;
        }


        public byte minRGBVal()
        {
            byte val = c0;
            val = val < c1 ? val : c1;
            val = val < c2 ? val : c2;
            return val;
        }

        public byte maxRGBVal()
        {
            byte val = c0;
            val = val > c1 ? val : c1;
            val = val > c2 ? val : c2;
            return val;
        }

        //Must be in RGB or BGR Format
        //Hue is in the range: [0, 360.0)
        public double computeHueFromRGB()
        {
            double hue = 0;
            int R, G, B;

            if (colorFormat == ImageColorFormat_e.RGB)
            {
                R = c0;
                G = c1;
                B = c2;

                int Min = minRGBVal();
                int Max = maxRGBVal();
                double C = (double)(Max - Min);
                if (C == 0)
                {
                    hue = 0.0;        //there is no correct Hue for this... so set to zero
                }
                else if (R == Max)
                {
                    hue = (double)(G - B) / C;
                    hue = hue < 0 ? 6.0 + hue : hue;    //ensures positive angles.
                }
                else if (G == Max)
                {
                    hue = ((double)(B - R) / C) + 2.0;
                }
                else
                {
                    hue = ((double)(R - G) / C) + 4.0;
                }
                //The above leaves Hue in the range of [0, 6.0)
                hue = 60.0 * hue;
            }
            return hue;
        }

        //Must be in RGB or BGR Format
        //Chroma is in the range: [0, 100.0]
        public double computeChromaFromRGB()
        {
            double chroma = 0;
            if (colorFormat == ImageColorFormat_e.RGB)
            {
                int c = maxRGBVal() - minRGBVal();
                chroma = (100.0 / 255.0) * (double)c;
            }
            return chroma;
        }


        //Must be in RGB or BGR Format
        //Value is in the range: [0, 100.0]
        public double computeHSV_V_FromRGB()
        {
            double V = 0;
            if (colorFormat == ImageColorFormat_e.RGB)
            {
                V = (100.0 / 255.0) * (double)maxRGBVal();
            }
            return V;
        }


        //Must be in RGB or BGR Format
        //Luminance is in the range: [0, 100.0]
        public double computeHSL_L_FromRGB()
        {
            double L = 0;
            if (colorFormat == ImageColorFormat_e.RGB)
            {
                L = 0.5 * (double)(maxRGBVal() - minRGBVal());
            }
            return L;
        }

        //Must be in RGB or BGR Format
        //Intensity is in the range: [0, 100.0]
        public double computeHSI_I_FromRGB()
        {
            double I = 0;
            if (colorFormat == ImageColorFormat_e.RGB)
            {
                I = (1.0 / 3.0) * (double)((int)c0 + (int)c1 + (int)c2);
            }
            return I;
        }


        //Change RGB or BGR to YCrCb format.
        //If already in that format.. noting happens.
        public PixelColorValue_t RGBToYCrCb()
        {
            PixelColorValue_t pcv = new PixelColorValue_t(this);
            double Y, Cr, Cb;
            if(colorFormat == ImageColorFormat_e.RGB)
            {
                Y = 0.257 * c0 + 0.504 * c1 + 0.098 * c2 + 16;
                Cr = 0.439 * c0 - 0.368 * c1 + -0.071 * c2 + 128;    // Cr image
                Cb = -0.148 * c0 - 0.2910 * c1 + 0.439 * c2 + 128;    // Cb image

                Y = Y < 0 ? 0 : Y > 255 ? 255 : Y;
                Cr = Cr < 0 ? 0 : Cr > 255 ? 255 : Cr;
                Cb = Cb < 0 ? 0 : Cb > 255 ? 255 : Cb;

                pcv.colorFormat = ImageColorFormat_e.YCrCb;
                pcv.c0 = (byte)Y;
                pcv.c1 = (byte)Cr;
                pcv.c2 = (byte)Cb;
            }
            return pcv;
        }

        public PixelColorValue_t YCrCbToRGB()
        {
            PixelColorValue_t pcv = new PixelColorValue_t(this);
            if (colorFormat == ImageColorFormat_e.YCrCb)
            {
                double r, g, b, Y, Cr, Cb;
                Y = c0 - 16.0;
                Cr = c1 - 128.0;
                Cb = c2 - 128.0;
                r = 1.16414435 * Y + 1.59578621 * Cr - 1.78889771e-3 * Cb;
                g = 1.16414435 * Y - 8.13482069e-1 * Cr - 3.9144276e-1 * Cb;
                b = 1.16414435 * Y - 1.24583948e-3 * Cr + 2.10782551 * Cb;

                r = r < 0 ? 0 : r > 255 ? 255 : r;
                g = g < 0 ? 0 : g > 255 ? 255 : g;
                b = b < 0 ? 0 : b > 255 ? 255 : b;

                pcv.colorFormat = ImageColorFormat_e.RGB;
                pcv.c0 = (byte)r;
                pcv.c1 = (byte)g;
                pcv.c2 = (byte)b;
            }
            return pcv;
        }


        //Change RGB or BGR to HSV format.
        //If already in that format.. noting happens.
        //All values are in the range of 0 to 240
        public PixelColorValue_t RGBToHSV()
        {
            double H, V, C;
            PixelColorValue_t pcv = new PixelColorValue_t(this);
            if(colorFormat == ImageColorFormat_e.RGB )
            {
                H = computeHueFromRGB();
                C = computeChromaFromRGB();
                V = computeHSV_V_FromRGB();

                pcv.c0 = (byte)((240.0 / 360.0) * H);
                pcv.c1 = 0;
                if(V > 0)
                {
                    pcv.c1 = (byte)(240.0 * C / V);
                }
                pcv.c2 = (byte)(2.40 * V);

                pcv.colorFormat = ImageColorFormat_e.HSV;
            }
            return pcv;
        }


        //Change RGB or BGR to HSV format.
        //If already in that format.. noting happens.
        //All values are in the range of 0 to 240
        public PixelColorValue_t RGBToHSL()
        {
            double H, V, C;
            PixelColorValue_t pcv = new PixelColorValue_t(this);
            if(colorFormat == ImageColorFormat_e.RGB )
            {
                H = computeHueFromRGB();
                C = computeChromaFromRGB();
                V = computeHSL_L_FromRGB();

                pcv.c0 = (byte)((240.0 / 360.0) * H);
                pcv.c1 = 0;
                if(V > 0 && V < 100.0)
                {

                    pcv.c1 = (byte)(240.0 * C / (100.0 - Math.Abs(2.0 * V - 100.0)));
                }
                pcv.c2 = (byte)(2.40 * V);

                pcv.colorFormat = ImageColorFormat_e.HSL;
            }
            return pcv;
        }


        //Change RGB or BGR to HSV format.
        //If already in that format.. noting happens.
        //All values are in the range of 0 to 240
        public PixelColorValue_t RGBToHSI()
        {
            double H, V, C;
            PixelColorValue_t pcv = new PixelColorValue_t(this);
            if(colorFormat == ImageColorFormat_e.RGB )
            {
                H = computeHueFromRGB();
                C = computeChromaFromRGB();
                V = computeHSI_I_FromRGB();
                double m = (100.0 / 255.0) * (double)minRGBVal();

                pcv.c0 = (byte)((240.0 / 360.0) * H);
                pcv.c1 = 0;
                if(V > 0)
                {
                    pcv.c1 = (byte)(240.0 * m / V);
                }
                pcv.c2 = (byte)(2.40 * V);

                pcv.colorFormat = ImageColorFormat_e.HSI;
            }
            return pcv;
        }


        //Change from current format to the new format.
        //Not all format changes are handled... in general conversion 
        //from RGB to other formats are handled... but few formats
        //back to RGB or other combinations are handle.  If the are not
        //handled, the new output format will contain zeros for the values.
        public PixelColorValue_t ChangeFormat(ImageColorFormat_e toFormat)
        {
            PixelColorValue_t pcv = new PixelColorValue_t();
            pcv.colorFormat = toFormat;
            switch (colorFormat)
            {
                case ImageColorFormat_e.RGB:
                    switch (toFormat)
                    {
                        case ImageColorFormat_e.RGB:
                            pcv.setRGBColor(c0, c1, c2);
                            break;
                        case ImageColorFormat_e.HSV:
                            pcv = RGBToHSV();
                            break;
                        case ImageColorFormat_e.HSL:
                            pcv = RGBToHSL();
                            break;
                        case ImageColorFormat_e.HSI:
                            pcv = RGBToHSI();
                            break;
                        case ImageColorFormat_e.YCrCb:
                            pcv = RGBToYCrCb();
                            break;
                    }
                    break;

                case ImageColorFormat_e.YCrCb:
                    if (toFormat == ImageColorFormat_e.RGB)
                    {
                        pcv = YCrCbToRGB();
                    }
                    break;
            }
            return pcv;
        }



    }
}
