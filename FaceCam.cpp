#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;

/* @ function main */
int main(int argc, char* argv[])
{
    // open the default camera
    cv::VideoCapture cap(0);

    // check if we succeeded
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera" << std::endl;
        return -1;
    }

    // face detection configuration
    cv::CascadeClassifier face_classifier;
    cv::CascadeClassifier face_classifier2;
    face_classifier.load("D:\\OpencvFile\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");

    cv::Mat frame;
    cv::Mat frame2;

    for (; ; ) {
        bool frame_valid = true;

        try {

            // get a new frame from webcam
            cap >> frame;
            cap >> frame2;
        }
        catch (cv::Exception& e) {

            std::cerr << "Exception occurred. Ignoring frame... " << e.err << std::endl;
            frame_valid = false;

        }

        if (frame_valid) {
            try {
                // convert captured frame to gray scale & equalize
                Scalar white(0, 0, 0);
                cv::Mat grayframe;
                cv::Mat grayframe2;
                cv::Mat face;

                cv::cvtColor(frame, grayframe, COLOR_BGR2GRAY);

                cv::cvtColor(frame2, grayframe2, COLOR_BGR2GRAY);
                threshold(grayframe2, frame2, 127, 255, THRESH_BINARY);

                Mat img_labels, stats, centroids;
                int numOfLables = connectedComponentsWithStats(face, img_labels, stats, centroids, 8, CV_32S);
                
                cv::Point myPoint;
                myPoint.x = 20;
                myPoint.y = 30;
                int myFontFace = 2;
                double myFontScale = 1.2;



                //for (int x = 0; x < face.cols; x++) {
                //    for (int y = 0; y < face.rows; y++) {
                //        if (face.at<Vec3b>(x, y) == Vec3b(255, 255, 255)) {
                //            cv::putText(frame, "Mask", myPoint, myFontFace, myFontScale, Scalar::all(255));
                //        }
                //    }
                //}
                cv::equalizeHist(grayframe, grayframe);

                // -------------------------------------------------------------
                // face detection routine

                // a vector array to store the face found
                std::vector<cv::Rect> faces;

                face_classifier.detectMultiScale(grayframe, faces,
                    1.1, // increase search scale by 10% each pass
                    2,   // merge groups of three detections
                    (1, 1),
                    cv::Size(30, 30)
                );

                // -------------------------------------------------------------
                // draw the results
                for (int i = 0; i < faces.size(); i++) {
                    cv::Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
                    cv::Point tr(faces[i].x, faces[i].y);
                    cv::rectangle(frame, lb, tr, cv::Scalar(0, 255, 0), 3, 4, 0);

                    frame2(faces[0]).copyTo(face);
                }

                for (int y = 0; y < img_labels.cols; y++) {
                    Vec3b* pixel = face.ptr<Vec3b>(y);
                    for (int x = 0; x < img_labels.rows; x++) 
                        {
                        if (pixel[y][x] >= 250) {
                           cv::putText(frame, "Mask", myPoint, myFontFace, myFontScale, Scalar::all(0));
                        }
                    }
                }
                cv::putText(frame, "Mask", myPoint, myFontFace, myFontScale, Scalar::all(0));
   /*             for (int x = 0; x < face.cols; x++) {
                    for (int y = 0; y < face.rows; y++) {
                        if (face.at<Vec3b>(x, y) == Vec3b()) {
                            cv::putText(frame, "mask", myPoint, myFontFace, myFontScale, Scalar::all(255));
                        }
                    }
                }*/
                //cv::putText(frame, "Mask", myPoint, myFontFace, myFontScale, Scalar::all(255));
                cv::imshow("마스크 구분", face);
                cv::imshow("이진화", frame2);
                cv::imshow("웹캠", frame);

            }
            catch (cv::Exception& e) {
                std::cerr << "Exception occurred. Ignoring frame... " << e.err << std::endl;
            }
        }

        if (cv::waitKey(30) >= 0) break;
    }

    // VideoCapture automatically deallocate camera object
    return 0;
}