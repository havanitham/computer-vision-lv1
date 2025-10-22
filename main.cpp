

#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main() {
    string imagePath = "D:/opencv_project/color.png";
    Mat img = imread(imagePath);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat blur;
    GaussianBlur(gray, blur, Size(9, 9), 0);
    Mat canny;
    Canny(blur, canny, 30, 150, 3);
    imshow("Original Image", img);
    imshow("Grayscale Image", gray);
    imshow("Blurred Image", blur);
    imshow("Canny Edges", canny);
    Mat result = img.clone();

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(canny, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    drawContours(result, contours, -1, Scalar(0, 255, 0), 2, LINE_8, hierarchy, 0);
    cout << "Pills detected: " << contours.size() << endl;
    imshow("Contours Drawn", result);
    waitKey(0);
    return 0;
}








