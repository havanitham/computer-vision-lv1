#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>  
using namespace cv;
using namespace std;

int main() {
    vector<string> imagePaths = {
        "D:/opencv_project/input_images/color.png",
        "D:/opencv_project/input_images/pink_pill.jpeg",
        "D:/opencv_project/input_images/orange_pill.jpeg"
    };

    for (int i = 0; i < imagePaths.size(); i++) {
        string imagePath = imagePaths[i];

        // read image
        Mat img = imread(imagePath);


        //converting the image to grayscale
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        
        //apply Gaussian blur 
        Mat blur;
        GaussianBlur(gray, blur, Size(9, 9), 0);

        //detect edges using Canny
        Mat canny;
        Canny(blur, canny, 30, 150, 3);

        //showing results
        imshow("Original Image", img);
        imshow("Grayscale Image", gray);
        imshow("Blurred Image", blur);
        imshow("Canny Edges", canny);

        Mat result = img.clone();

        //find contours 
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(canny, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        //draw the contours on the image
        drawContours(result, contours, -1, Scalar(0, 255, 0), 2, LINE_8, hierarchy, 0);

        cout << "Pills detected: " << contours.size() << endl;

        imshow("Contours Drawn", result);

        waitKey(0);
        destroyAllWindows(); 
    }

    return 0;
}

