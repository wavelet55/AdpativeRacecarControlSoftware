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


#ifndef VIDERE_DEV_FACEPOSEDETECTOR_H
#define VIDERE_DEV_FACEPOSEDETECTOR_H

//The face Pose Detector uses a libary that is painfully slow
//to compile ... so only enable this if really needed/used
//#define FACEPOSEDETECTORENABLED

#ifdef FACEPOSEDETECTORENABLED

#include "CommonImageProcTypesDefs.h"
#include <opencv2/core.hpp>
#include <vector>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <iostream>

namespace ImageProcLibsNS
{

    class FacePoseDetector
    {
    private:
        //Logging System
        log4cxx::LoggerPtr log4cpp_;

        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.
        dlib::frontal_face_detector _faceDetector;

        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        dlib::shape_predictor _shapePredictor;


        bool _faceDetectorLoaded = false;
        bool _shapePredictorLoaded = false;

        //Last Image
        cv::Mat *lastInpImage = nullptr;

    public:
        // Now tell the face detector to give us a list of bounding boxes
        // around all the faces in the image.
        std::vector<dlib::rectangle> FaceBoundingBoxes;

        // Now we will go ask the shape_predictor to tell us the pose of
        // each face we detected.
        std::vector<dlib::full_object_detection> FaceShapes;

        std::string FaceLandmarkDataFilename = "shape_predictor_68_face_landmarks.dat";

    public:


        FacePoseDetector();

        ~FacePoseDetector()
        {
            Close();
        }

        //Create a face-detector
        bool Initialize();

        //Release all resources.
        void Close()
        {
            releaseResources();
        }

        //Release local resources and temporary objects
        //Does not close out the FacePoseDetector... assumes
        //the FacePoseDetector may be used again.
        void releaseResources();

        //Detect Faces and return the number of faces found.
        int DetectFaces(cv::Mat *imgInpBGR);

        bool AddFaceFeaturesToLastImage(cv::Mat &outImg);

        //Add Face Features to Image
        void AddFaceFeaturesToImage(cv::Mat &image);

    };

}

#endif //FACEPOSEDETECTORENABLED

#endif //VIDERE_DEV_FACEPOSEDETECTOR_H
