/* ****************************************************************
 * Blob Target Detector
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Sept 2016
 *
 * Developed under contract for:
 * Academy Center for UAS Research
 * Department of Electrical and Computer Engineering
 * HQ USAFA/DFEC
 * 2354 Fairchild Drive
 * USAF Academy, CO 80840-6236
  *******************************************************************/

#ifndef VIDERE_DEV_BLOBDETECTORFIXEDPARAMETERS_H
#define VIDERE_DEV_BLOBDETECTORFIXEDPARAMETERS_H

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "PixelColorValue.h"

using namespace ImageProcLibsNS;

#define BDFP_PI              3.141592
#define DEBUG                0            // Display debugging comments
#define IMAGE_DISPLAY        0            // Display each received image
#define IMAGE_SAVE           0            // Save each received image

// Boat size (only used with background modeling.  Not needed for this project)
#define SIZE_BOAT_LOW            12        // 320x240 resolution @ 250m
#define SIZE_BOAT_HIGH            200        // 320x240 resolution @ 250m
#define SIZE_BOAT_DESIRED        30        // 320x240 resolution @ 250m
#define BOAT_CIRCULAR_RATE        0.17

#define MAX_NUMBER_TARGETS_PER_IMAGE   10

// Color patch size for target type 1
#define SIZE_TYPE1_PATCH_LOW            20        // 320x240 resolution @ 250m
#define SIZE_TYPE1_PATCH_HIGH            300        // 320x240 resolution @ 250m
#define SIZE_TYPE1_PATCH_DESIRED        100    // 320x240 resolution @ 250m
#define PATCH_TYPE1_CIRCULAR_RATE        0.050

// Color patch size for target type 2
// The range of path size is very small. In this way, We don't want to detect type 2
#define SIZE_TYPE2_PATCH_LOW            100        // 320x240 resolution @ 250m
#define SIZE_TYPE2_PATCH_HIGH            110        // 320x240 resolution @ 250m
#define SIZE_TYPE2_PATCH_DESIRED        105        // 320x240 resolution @ 250m
#define PATCH_TYPE2_CIRCULAR_RATE        0.17

#define DISTANCE_BETWEEN_BOATS    30        // Minimum distance between adjacent boats

#define ALT_UAV_LOW            1        // Minimum UAV altitude
#define ALT_UAV_HIGH           10000    // Maximum UAV altitude

#define PITCH_LOW            -1.05        // -60 degrees (radians)
#define PITCH_HIGH            1.05        // +60 degrees (radians)
#define ROLL_LOW             -1.05        // -60 degrees (radians)
#define ROLL_HIGH             1.05        // 60 degrees (radians)

#define TASE_HIGH            -0.2618        // -PI/12

#define MAXNUM_TARGET        4            // Maximum number of targets

//
// Background modeling
//
#define numBGSample        100        // Color histogram sampling range
#define sizeMorphFilter    10        // Radius of the morphological filter
#define proportionBG    8        // Proportion of size from the actual to the pixel

//
// Uncertainty information
//
#define uncertaintyDetection            0.95    // Simulation only
#define uncertaintyX                    25.0    // x-directional localization uncertainty
#define uncertaintyY                    25.0    // y-directional localization uncertainty
#define uncertaintyVelX                    10.0    // x-directional velocity uncertainty
#define uncertaintyVelY                    10.0    // y-directional velocity uncertainty
#define uncertaintyAngleDeg                7.0        // degree
#define uncertaintyOrientationDeg        20.0    // degree



namespace CommonBlobTargetDetectorNS
{

// Calibration
    struct calibInfo
    {
        double focalLength;
        double centerX;
        double centerY;
        double scaleFactor;

        double distCoeff[5];
        double intM[9];
        double Rot[9];
        double Trn[3];
    };

// x,y coordiate struture
    struct xyCoord_t
    {
        double x;
        double y;
    };

// Target detection code
    enum TargetCode_e
    {
        NoTarget = 0,
        TargetType1 = 1,
        TargetType2 = 2,
        UnknownTarget = 3
    };

// Target color code
    enum TargetColor_e
    {
        RED = 0,
        YELLOW = 1,
        BLUE = 2,
        THERMAL = 3,
        UNKNOWN = 4
    };



/********Sensor Type ****(See Global Defines)************
    enum SensorType_e
    {
        EO = 0,
        IR = 1
    };
************************************/

// Image processing status
    enum ImageProcessingStatus_e
    {
        NoTargetDetected = 0,
        TargetType1DetectedOnly = 1,
        TargetType2DetectedOnly = 2,
        UnknownTargetDetectedOnly = 3,
        Type1AndType2Detected = 4,
        Type1AndUnknownDetected = 5,
        Type2AndUnknownDetected = 6,
        AllDetected = 7,
// WARNING: ONLY ERROR CODES THAT RESULT IN AN INVALID IMAGE SHOULD EXIST AT VALUES 10 and ABOVE
                AltitudeOutOfRange = 10,
        PitchAngleOutOfRange = 11,
        RollAngleOutOfRange = 12,
        TASEAngleOutOfRange = 13,

        BadImageData = 20,

        WrongColorCode = 30,

        OptionMismatch = 40,

        Unset = 100            // Used by HOPS to indicate that it has not set this status to any value yet.
    };

// Image processing mode
// sim -- use simulation data
// img -- use stored images
// cam -- use real camera images
    enum Mode_e
    {
        sim = 0,
        img = 1,
        cam = 2
    };

/// <summary>
/// A 2x2 Covariance Matrix type.
/// This is generated as a structure because it is
/// easier to marshal across an interface.
/// Note: Do not change the Structure ordering nor place
/// any other parameters in front of the cvnm parameters.
/// There is a complementary structure in image processing.
/// </summary>

    union CovMatrix_t
    {
        struct
        {
            double cv11;
            double cv12;
            double cv13;
            double cv14;
            double cv21;
            double cv22;
            double cv23;
            double cv24;
            double cv31;
            double cv32;
            double cv33;
            double cv34;
            double cv41;
            double cv42;
            double cv43;
            double cv44;
        } vals;
        double mtx[4][4];
    };


/// <summary>
/// The Quad Corners type is a set of four
/// xyCoord_t corners which can be used to define the
/// corners of an image on the ground or other "retangular"
/// object.  The structure is defined as a set of four
/// xyCoord_t corner structures rather than an array of
/// xyCoord_t corners.  This was done on purpose to make
/// copies and Marshaling of the object more efficient.
///
/// Note: Do not change the Structure ordering nor place
/// any other parameters in front of the cvnm parameters.
/// There is a complementary structure in image processing.
///
/// If the structure is transformed to Lat/Lon
/// then cn.x is the latitude in radians
/// and cy.y is the longitude in radians.
/// </summary>
    union QuadCorners_t
    {
        struct
        {
            xyCoord_t c1;
            xyCoord_t c2;
            xyCoord_t c3;
            xyCoord_t c4;
        } vals;
        xyCoord_t mtx[4];
    };


/// <summary>
/// Target Results is a data structure that contains
/// information for a single target within an image.
/// </summary>
    struct TargetResults_t
    {
        char latLonStorage;
        int targetCode;

        xyCoord_t TargetPixelLocation;
        xyCoord_t TargetGroundXYLocation;
        double TargetAzimuthRadians;
        double TargetElevationRadians;
        double TargetOrientationRadians;        //Track:14APR14-01

        CovMatrix_t TargetCovariance;
    };

/// <summary>
/// ImageProcResults is a data structure that contains
/// an array of target results.  There will be one
/// target result for each target found in an image.
/// The maximum number of targets must be consistant with
/// the marshaling definitions.  The Max number of targets is
/// passed across to the image processing module.
/// </summary>
    struct ImageProcResultsCpp_t
    {
        //The MaxNumberOfTargets is the size of the
        //TargetResults Array.  Do not change this number
        //internally.
        int MaxNumberOfTargets;

        int ImageProcStatus;            //Use values from: ImageProcessingStatus_e
        int NumberOfTargetType1;
        int NumberOfTargetType2;
        int NumberOfUnknownTargets;

        QuadCorners_t ImageGroundXYCorners;

        TargetResults_t TargetResults[4];
    };


/// <summary>
/// The UAV Intertial States is the location an
/// attitude of the UAV required by Image Processing.
/// </summary>
    struct UAVInertialStates_t
    {
        /// <summary>
        /// The UAV position in the HOPS XY Coordinate plane.
        /// </summary>
        xyCoord_t UAV_Position;

        /// <summary>
        /// The UAV Altitude above ground level in meters.
        /// </summary>
        double AltitudeAGL;

        /// <summary>
        /// The UAV yaw pitch and roll angles in radians.
        /// </summary>
        double yaw;
        double pitch;
        double roll;
    };


/// Target Parameters used to define the Type1 or
/// Type2 target.
    struct TargetParameters_t
    {
        /// <summary>
        /// Target Length in meters
        /// </summary>
        double Length;

        /// <summary>
        /// Target Width in meters
        /// </summary>
        double Width;

        /// <summary>
        /// Target Color
        /// </summary>
        int color;
    };

// Color thresholds
    struct Color_Thresholds
    {
        int COLOR_RED_CR_TOP;        // Red
        int COLOR_RED_CR_DOWN;
        int COLOR_RED_CB_TOP;
        int COLOR_RED_CB_DOWN;

        int COLOR_YELLOW_CR_TOP;    // Yellow
        int COLOR_YELLOW_CR_DOWN;
        int COLOR_YELLOW_CB_TOP;
        int COLOR_YELLOW_CB_DOWN;

        int COLOR_BLUE_CR_TOP;        // Blue
        int COLOR_BLUE_CR_DOWN;
        int COLOR_BLUE_CB_TOP;
        int COLOR_BLUE_CB_DOWN;

        int IR_TOP;                    //IR
        int IR_DOWN;
    };

    struct ColorThresholds_t
    {
        int CrTop;      //Use also for IR Top
        int CrDown;     //Use also for IR Down
        int CbTop;
        int CbDown;

        void ColorThreshold_t(int crTop, int crDown, int cbTop, int cbDown)
        {
            CrTop = crTop;
            CrDown = crDown;
            CbTop = cbTop;
            CbDown = cbDown;
        }

        void Clear()
        {
            CrTop = 0;
            CrDown = 0;
            CbTop = 0;
            CbDown = 0;
        }

        void SetColorThresholds(int crTop, int crDown, int cbTop, int cbDown)
        {
            CrTop = crTop;
            CrDown = crDown;
            CbTop = cbTop;
            CbDown = cbDown;
        }

    };



    // Image mask thresholds (value in pixels, with upper-left corner is (1,1)
    struct ImageMaskRect_t
    {
        int MASK_PIXEL_X_MIN;
        int MASK_PIXEL_X_MAX;
        int MASK_PIXEL_Y_MIN;
        int MASK_PIXEL_Y_MAX;
    };

// TASE information
    struct TASE_info
    {
        double azimuth;        // azimuth (rad)
        double elevation;    // elevation (rad)
    };

// Image processing options
    struct Options_t
    {
        int zoomIn;                    // Zoomed in or not?
        int backgroundModeling;        // Using background modeling data?
        int sunlightReflection;        // Removing sunlight reflected areas?
        int wheelRemoval;            // Ignoring wheel occlusion?

        int targetType1;            // Do we detection Type1 targets?
        int targetType2;            // Do we detection Type2 targets?
        int unknownTarget;            // Do we detection unknown targets?

        double BGMode[9];            // For 3 modes, nR mean, nG mean, and its standard deviation
        double sunOrientation[2];    // Azimuth and elevation

        int imageMask;                // Mask out a portion of the image from processing?
        int markTarget;                // Mark found targets in red on image if set
    };


// Simulator target data
    struct SimData_t
    {
        int simNumType1;
        int simNumType2;
        int simNumUnknown;

        double targetType1x[4];
        double targetType1y[4];

        double targetType2x[4];
        double targetType2y[4];

        double targetUx[4];
        double targetUy[4];
    };


    struct BlobDetectorParameters_t
    {
        //OpenCV Simple Blob Detector:
        cv::SimpleBlobDetector::Params BlobDetParams;

        bool BlobDetParamChanged = true;

        //Pixel Color Range Settings
        PixelColorValue_t BlobMinColorValue;
        PixelColorValue_t BlobMaxColorValue;

        // Zoomed in or not?
        bool IsZoomedIn = false;

        // Using background modeling data?
        bool BackgroundModelingEnabled = false;

        // Removing sunlight reflected areas?
        bool SunlightReflectionEnabled = false;

        // Ignoring wheel occlusion?
        bool WheelRemovalEnabled = false;

        // Mask out a portion of the image from processing?
        bool ImageMaskEnabled = false;

        bool DetectType1Targets = true;
        bool DetectType2Targets = true;
        bool DetectUnknownTargets = true;

        //Set to true for IR Sensors.
        //Otherwise an RGB EO sensor is assumed.
        bool IsIRSensor = false;

        // For 3 modes, nR mean, nG mean, and its standard deviation
        double BGMode[9];

        double SunOrientation[2];    // Azimuth and elevation

        //SensorType_e SensorType = SensorType_e::EO;


        bool UseGausianFilter = true;
        int GausianFilterBorderType = (int)cv::BorderTypes::BORDER_DEFAULT;

        //The GausianFilterKernalSize must be an odd number or zero
        //If zero they are computed from Sigma
        int GausianFilterKernalSize= 5;   //Use same for X & Y
        double GausianFilterSigma = 0;   //Use same for X & Y        PixelColorValue_t BlobMinColorValue;

        int getGausianFilterKernalSize() {return GausianFilterKernalSize;}
        void setGausianFilterKernalSize(int val)
        {
            GausianFilterKernalSize = val < 0 ? 0 : val > 100 ? 100 : val;
            //Force it to be an odd number.
            GausianFilterKernalSize = GausianFilterKernalSize >> 1;
            GausianFilterKernalSize = GausianFilterKernalSize << 1;
            GausianFilterKernalSize += 1;
            BlobDetParamChanged = true;
        }

        double getGausianFilterSigma() {return GausianFilterSigma;}
        void setGausianFilterSigma(double val)
        {
            GausianFilterSigma = val < 0 ? 0 : val > 100.0 ? 100.0 : val;
            BlobDetParamChanged = true;
        }


        //1.  Convert the source image to binary images by applying thresholding with several thresholds from
        //minThreshold (inclusive) to maxThreshold (exclusive) with distance thresholdStep between
        //neighboring thresholds.
        double getBlobDetectorMinThreshold() {return BlobDetParams.minThreshold;}
        void setBlobDetectorMinThreshold(double val)
        {
            BlobDetParams.minThreshold = (float)(val < 0 ? 0 : val > 256 ? 256 : val);
            BlobDetParamChanged = true;
        }
        double getBlobDetectorMaxThreshold() {return BlobDetParams.maxThreshold;}
        void setBlobDetectorMaxThreshold(double val)
        {
            BlobDetParams.maxThreshold = (float)(val < 0 ? 0 : val > 256 ? 256 : val);
            BlobDetParamChanged = true;
        }
        double getBlobDetectorThresholdStep() {return BlobDetParams.thresholdStep;}
        void setBlobDetectorThresholdStep(double val)
        {
            BlobDetParams.thresholdStep = (float)(val < 0 ? 0 : val > 256 ? 256 : val);
            BlobDetParamChanged = true;
        }
        double getBlobDetectorMinDistBetweenBlobs() {return BlobDetParams.minDistBetweenBlobs;}
        void setBlobDetectorMinDistBetweenBlobs(double val)
        {
            BlobDetParams.minDistBetweenBlobs = (float)(val < 0 ? 0 : val > 10000 ? 10000 : val);
            BlobDetParamChanged = true;
        }

        bool getBlobDetectorFilterByColor() {return BlobDetParams.filterByColor;}
        void setBlobDetectorFilterByColor(bool val)
        {
            BlobDetParams.filterByColor = val;
            BlobDetParamChanged = true;
        }
        u_char getBlobDetectorBlobColor() {return BlobDetParams.blobColor;}
        void setBlobDetectorBlobColor(u_char val)
        {
            BlobDetParams.blobColor = val;
            BlobDetParamChanged = true;
        }

        bool getBlobDetectorFilterByArea() {return BlobDetParams.filterByArea;}
        void setBlobDetectorFilterByArea(bool val)
        {
            BlobDetParams.filterByArea = val;
            BlobDetParamChanged = true;
        }
        double getBlobDetectorMinArea() {return BlobDetParams.minArea;}
        void setBlobDetectorMinArea(double val)
        {
            BlobDetParams.minArea = (float)(val < 0 ? 0 : val > 1000000 ? 1000000 : val);
            BlobDetParamChanged = true;
        }
        double getBlobDetectorMaxArea() {return BlobDetParams.maxArea;}
        void setBlobDetectorMaxArea(double val)
        {
            BlobDetParams.maxArea = (float)(val < 0 ? 0 : val > 1000000 ? 1000000 : val);
            BlobDetParamChanged = true;
        }


        bool getBlobDetectorFilterByCircularity() {return BlobDetParams.filterByCircularity;}
        void setBlobDetectorFilterByCircularity(bool val)
        {
            BlobDetParams.filterByCircularity = val;
            BlobDetParamChanged = true;
        }
        double getBlobDetectorMinCircularity() {return BlobDetParams.minCircularity;}
        void setBlobDetectorMinCircularity(double val)
        {
            BlobDetParams.minCircularity = (float)(val < 0 ? 0 : val > 1000000 ? 1000000 : val);
            BlobDetParamChanged = true;
        }
        double getBlobDetectorMaxCircularity() {return BlobDetParams.maxCircularity;}
        void setBlobDetectorMaxCircularity(double val)
        {
            BlobDetParams.maxCircularity = (float)(val < 0 ? 0 : val > 1000000 ? 1000000 : val);
            BlobDetParamChanged = true;
        }

        bool getBlobDetectorFilterByConvexity() {return BlobDetParams.filterByConvexity;}
        void setBlobDetectorFilterByConvexity(bool val)
        {
            BlobDetParams.filterByConvexity = val;
            BlobDetParamChanged = true;
        }
        double getBlobDetectorMinConvexity() {return BlobDetParams.minConvexity;}
        void setBlobDetectorMinConvexity(double val)
        {
            BlobDetParams.minConvexity = (float)(val < 0 ? 0 : val > 1000000 ? 1000000 : val);
            BlobDetParamChanged = true;
        }
        double getBlobDetectorMaxConvexity() {return BlobDetParams.maxConvexity;}
        void setBlobDetectorMaxConvexity(double val)
        {
            BlobDetParams.maxConvexity = (float)(val < 0 ? 0 : val > 1000000 ? 1000000 : val);
            BlobDetParamChanged = true;
        }

        bool getBlobDetectorFilterByInertia() {return BlobDetParams.filterByInertia;}
        void setBlobDetectorFilterByInertia(bool val)
        {
            BlobDetParams.filterByInertia = val;
            BlobDetParamChanged = true;
        }
        double getBlobDetectorMinInertiaRatio() {return BlobDetParams.minInertiaRatio;}
        void setBlobDetectorMinInertiaRatio(double val)
        {
            BlobDetParams.minInertiaRatio = (float)(val < 0 ? 0 : val > 1000000 ? 1000000 : val);
            BlobDetParamChanged = true;
        }
        double getBlobDetectorMaxInertiaRatio() {return BlobDetParams.maxInertiaRatio;}
        void setBlobDetectorMaxInertiaRatio(double val)
        {
            BlobDetParams.maxInertiaRatio = (float)(val < 0 ? 0 : val > 1000000 ? 1000000 : val);
            BlobDetParamChanged = true;
        }


        void setDefaultParameters()
        {
            UseGausianFilter = true;
            //The GausianFilterKernalSize must be an odd number or zero
            //If zero they are computed from Sigma
            GausianFilterKernalSize= 5;   //Use same for X & Y
            GausianFilterSigma = 0;   //Use same for X & Y
            GausianFilterBorderType = (int)cv::BorderTypes::BORDER_DEFAULT;

            BlobDetParams.minThreshold = 10.0f;
            BlobDetParams.maxThreshold = 200.0f;
            BlobDetParams.thresholdStep = 150.0f;
            BlobDetParams.minDistBetweenBlobs = 100.0f;

            BlobDetParams.filterByColor = true;
            BlobDetParams.blobColor = 255;   //Extracts a white blob

            BlobDetParams.filterByArea = true;
            BlobDetParams.minArea = 100.0f;
            BlobDetParams.maxArea = 2500.0f;

            BlobDetParams.filterByCircularity = false;
            BlobDetParams.minCircularity = 0.1f;
            BlobDetParams.maxCircularity = 1.0f;

            BlobDetParams.filterByConvexity = false;
            BlobDetParams.minConvexity = 0.75f;
            BlobDetParams.maxConvexity = 1.0f;

            BlobDetParams.filterByInertia = false;
            BlobDetParams.minInertiaRatio = 0.1f;
            BlobDetParams.maxInertiaRatio = 1000000.0f;
        }

        BlobDetectorParameters_t()
        {
            setDefaultParameters();
        }

        BlobDetectorParameters_t(const BlobDetectorParameters_t &bdps)
        {
            *this = bdps;
        }

    };

}

#endif //VIDERE_DEV_BLOBDETECTORFIXEDPARAMETERS_H
