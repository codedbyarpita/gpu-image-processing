#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("../images/input.jpg", IMREAD_GRAYSCALE);
    Mat output = img.clone();

    auto start = chrono::high_resolution_clock::now();

    for(int y = 1; y < img.rows-1; y++) {
        for(int x = 1; x < img.cols-1; x++) {
            int sum = 0;
            for(int dy = -1; dy <= 1; dy++) {
                for(int dx = -1; dx <= 1; dx++) {
                    sum += img.at<uchar>(y+dy, x+dx);
                }
            }
            output.at<uchar>(y,x) = sum / 9;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "CPU Time: " 
         << chrono::duration<double>(end-start).count() 
         << " sec\n";

    imwrite("../outputs/cpu.jpg", output);
    return 0;
}