/* ****************************************************************
 * DireenTech Inc.  (www.direentech.com)
 * Athr: Harry Direen PhD
 * Date: Feb. 2018
 *
 * Uses DLib Face detector and Pose algorithm... reference the
 * example pgm:  face_landmark_detection_ex.cpp
 * Notes from:  face_landmark_detection_ex
 *
 * This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.

    The face detector we use is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
       One Millisecond Face Alignment with an Ensemble of Regression Trees by
       Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset (see
    https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
       C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
       300 faces In-the-wild challenge: Database and results.
       Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
    You can get the trained model file from:
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
    Note that the license for the iBUG 300-W dataset excludes commercial use.
    So you should contact Imperial College London to find out if it's OK for
    you to use this model file in a commercial product.


    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

 * This Face detector relies on the above mentioned shape_predictor_68_face_landmarks.dat
 * which is not included in the git repository due to it's size.
 *
  *******************************************************************/

#ifdef FACEPOSEDETECTORENABLED


#include "FacePoseDetector.h"
#include <dlib/opencv/cv_image.h>
#include <opencv2/imgproc.hpp>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

using namespace dlib;
using namespace std;

namespace ImageProcLibsNS
{

    FacePoseDetector::FacePoseDetector()
        : _faceDetector(), _shapePredictor(),
          FaceLandmarkDataFilename("shape_predictor_68_face_landmarks.dat")
    {
        _faceDetectorLoaded = false;
        _shapePredictorLoaded = false;

    }

    bool FacePoseDetector::Initialize()
    {
        bool error = true;
        try
        {
            // We need a face detector.  We will use this to get bounding boxes for
            // each face in an image.
            //_faceDetector.
            _faceDetector = get_frontal_face_detector();

            // And we also need a shape_predictor.  This is the tool that will predict face
            // landmark positions given an image and face bounding box.  Here we are just
            // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
            // as a command line argument.
            dlib::deserialize(FaceLandmarkDataFilename) >> _shapePredictor;

            _faceDetectorLoaded = true;
            _shapePredictorLoaded = true;
        }
        catch (std::exception &e)
        {
            LOGERROR("FacePoseDetector Initialize: Exception: " << e.what());
            _faceDetectorLoaded = false;
            _shapePredictorLoaded = false;
            error = true;
        }
        return error;
    }

    //Release local resources and temporary objects
    //Does not close out the FacePoseDetector... assumes
    //the FacePoseDetector may be used again.
    void FacePoseDetector::releaseResources()
    {
        FaceBoundingBoxes.clear();
        FaceBoundingBoxes.shrink_to_fit();
        FaceShapes.clear();
        FaceShapes.shrink_to_fit();
    }


    //Detect Faces and return the number of faces found.
    int FacePoseDetector::DetectFaces(cv::Mat *imgInpBGR)
    {
        int numFaces = 0;
        if(!_faceDetectorLoaded || !_shapePredictorLoaded)
        {
            if( Initialize() )
            {
                return 0;
            }
        }

        lastInpImage = imgInpBGR;

        //Convert OpenCV image to dlib::array2d
        dlib::array2d<bgr_pixel> dlibImage;
        dlib::assign_image(dlibImage, dlib::cv_image<bgr_pixel>(*imgInpBGR));

        // Make the image larger so we can detect small faces.
        //dlib::pyramid_up(dlibImage);
        //*lastInpImage = dlib::toMat(dlibImage);

        //consider using the following so we don't change the input image:
        //void pyramid_up (const image_type1& in_img,image_type2& out_img,const pyramid_type& pyrj);

        // Now tell the face detector to give us a list of bounding boxes
        // around all the faces in the image.
        FaceBoundingBoxes = _faceDetector(dlibImage);
        numFaces = FaceBoundingBoxes.size();


        // Now we will go ask the shape_predictor to tell us the pose of
        // each face we detected.
        FaceShapes.clear();
        for (unsigned long j = 0; j < numFaces; ++j)
        {
            full_object_detection shape = _shapePredictor(dlibImage, FaceBoundingBoxes[j]);
            cout << "number of parts: "<< shape.num_parts() << endl;
            cout << "pixel position of first part:  " << shape.part(0) << endl;
            cout << "pixel position of second part: " << shape.part(1) << endl;
            // You get the idea, you can get all the face part locations if
            // you want them.  Here we just store them in shapes so we can
            // put them on the screen.
            FaceShapes.push_back(shape);
        }

        return numFaces;
    }

    cv::Point tocvPoint(const dlib::point &pt)
    {
        cv::Point cvPt;
        cvPt.x = (int)pt.x();
        cvPt.y = (int)pt.y();
        return cvPt;
    }

    bool FacePoseDetector::AddFaceFeaturesToLastImage(cv::Mat &outImg)
    {
        bool imageAvailable = false;
        if(lastInpImage != nullptr)
        {
            if(lastInpImage->rows > 0 && lastInpImage->cols > 0)
            {
                lastInpImage->copyTo(outImg);
                AddFaceFeaturesToImage(outImg);
                imageAvailable = true;
            }
        }
        return imageAvailable;
    }

    //Add Face Features to Image
    void FacePoseDetector::AddFaceFeaturesToImage(cv::Mat &image)
    {
        //dlib::render_face_detections(FaceShapes);
        //ToDo:  Pass in color value.
        cv::Scalar color(0, 255, 0);  //Blue Green Red color Values
        cv::Point p1, p2;

        for (unsigned long i = 0; i < FaceShapes.size(); ++i)
        {
            const full_object_detection& d = FaceShapes[i];

            if (d.num_parts() == 5)  //Either 5 or 68 parts...
            {
                cv::line(image, tocvPoint(d.part(0)), tocvPoint(d.part(1)), color);
                cv::line(image, tocvPoint(d.part(1)), tocvPoint(d.part(4)), color);
                cv::line(image, tocvPoint(d.part(4)), tocvPoint(d.part(3)), color);
                cv::line(image, tocvPoint(d.part(3)), tocvPoint(d.part(2)), color);
            }
            else if(d.num_parts() == 68)   //Must
            {
                // Around Chin. Ear to Ear
                for (unsigned long i = 1; i <= 16; ++i)
                    cv::line(image, tocvPoint(d.part(i)), tocvPoint(d.part(i - 1)), color);

                // Line on top of nose
                for (unsigned long i = 28; i <= 30; ++i)
                    cv::line(image, tocvPoint(d.part(i)), tocvPoint(d.part(i - 1)), color);

                // left eyebrow
                for (unsigned long i = 18; i <= 21; ++i)
                    cv::line(image, tocvPoint(d.part(i)), tocvPoint(d.part(i - 1)), color);
                // Right eyebrow
                for (unsigned long i = 23; i <= 26; ++i)
                    cv::line(image, tocvPoint(d.part(i)), tocvPoint(d.part(i - 1)), color);
                // Bottom part of the nose
                for (unsigned long i = 31; i <= 35; ++i)
                    cv::line(image, tocvPoint(d.part(i)), tocvPoint(d.part(i - 1)), color);

                // Line from the nose to the bottom part above
                cv::line(image, tocvPoint(d.part(30)), tocvPoint(d.part(35)), color);

                // Left eye
                for (unsigned long i = 37; i <= 41; ++i)
                    cv::line(image, tocvPoint(d.part(i)), tocvPoint(d.part(i - 1)), color);
                cv::line(image, tocvPoint(d.part(36)), tocvPoint(d.part(41)), color);

                // Right eye
                for (unsigned long i = 43; i <= 47; ++i)
                    cv::line(image, tocvPoint(d.part(i)), tocvPoint(d.part(i - 1)), color);
                cv::line(image, tocvPoint(d.part(42)), tocvPoint(d.part(47)), color);

                // Lips outer part
                for (unsigned long i = 49; i <= 59; ++i)
                    cv::line(image, tocvPoint(d.part(i)), tocvPoint(d.part(i - 1)), color);
                cv::line(image, tocvPoint(d.part(48)), tocvPoint(d.part(59)), color);

                // Lips inside part
                for (unsigned long i = 61; i <= 67; ++i)
                    cv::line(image, tocvPoint(d.part(i)), tocvPoint(d.part(i - 1)), color);
                cv::line(image, tocvPoint(d.part(60)), tocvPoint(d.part(67)), color);
            }
        }
    }

}

#endif  // FACEPOSEDETECTORENABLED
