#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace dlib;
using namespace cv;
using namespace cv::ml;

int descriptor_size2;
int k=0;

HOGDescriptor hog(
        Size(240,320), //winSize
        Size(8,8), //blocksize
        Size(8,8), //blockStride,
        Size(4,4), //cellSize,
                9, //nbins,
                0, //derivAper,
                4, //winSigma,
                1, //histogramNormType,
                0.2, //L2HysThresh,
                1,//gammal correction,
                64,//nlevels=64
                1);//signedGradient

int CreateHOG(std::vector<std::vector<float> > &HOG, std::vector<Mat> &Images)
{
    for(int y = 0; y < Images.size(); y++)
    {
        std::vector<float> descriptors;
        hog.compute(Images[y],descriptors);
        HOG.push_back(descriptors);
    }
    return HOG[0].size();
}

void ConvertVectortoMatrix(std::vector<std::vector<float> > &HOG, Mat &samples)
{
    int descriptor_size = HOG[0].size();

    for(int i = 0; i < HOG.size(); i++)
        for(int j = 0; j < descriptor_size; j++)
            samples.at<float>(i,j) = HOG[i][j];
}


void svmPredict(Ptr<SVM> svm, Mat &testResponse, Mat &testMat )
{
    svm->predict(testMat, testResponse);
}


int main()
{
    try
    {

        string filename = "./images/IMG_20180828_114339.jpg";

        dlib::array2d<dlib::bgr_pixel> img_rgb;

        typedef scan_fhog_pyramid<pyramid_down<3> > image_scanner_type;
        image_scanner_type scanner;
        //unsigned long width=80, height=80;
        //scanner.set_detection_window_size(width, height);

        dlib::object_detector<image_scanner_type> detector;
        deserialize("./object_detector.svm")>>detector;
        Ptr<SVM> savedModel = StatModel::load<SVM>("./model/ClassModel2.yml");

        dlib::load_image(img_rgb, filename);
        std::vector<dlib::rectangle> dets = detector(img_rgb);
        cout<<dets.size()<<endl;

        Mat testImage = dlib::toMat(img_rgb);
        Mat testImage2 = dlib::toMat(img_rgb);

        for(unsigned long i=0; i<dets.size(); i++)
        {
            string img_name = "./segments/" + to_string(k) + ".jpg";
            int left=dets[i].left(), top=dets[i].top(), right=dets[i].right(), bottom=dets[i].bottom();
            //cout<<dets[i].left()<<' '<<dets[i].top()<<' '<<dets[i].right()<<' '<<dets[i].bottom()<<endl;
            cv::rectangle(testImage, Rect(left, top, right-left, bottom-top), Scalar(0,255,0), 3);

            Mat crop = testImage2(Rect(left, top, right-left, bottom-top));
            resize(crop, crop, Size(240, 320));
            imwrite(img_name, crop);

            std::vector<Mat> testImageArray;
            testImageArray.push_back(crop);

            std::vector<std::vector<float> > testHOGArray;
            descriptor_size2 = CreateHOG(testHOGArray, testImageArray);

            Mat testSample(testHOGArray.size(), descriptor_size2, CV_32FC1);
            ConvertVectortoMatrix(testHOGArray, testSample);

            Mat pred;
            svmPredict(savedModel, pred, testSample);

            cout << "Prediction : " << pred.at<float>(0,0) << endl;
            if(pred.at<float>(0,0)==0)
                cv::rectangle(testImage2, Rect(left, top, right-left, bottom-top), Scalar(255,0,0), 3);
            else
                cv::rectangle(testImage2, Rect(left, top, right-left, bottom-top), Scalar(0,0,255), 3);
            //imshow("Test Image", testImage);
            //waitKey(0);
            k++;
        }
        imwrite("./result.jpg",testImage2);
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }

}





