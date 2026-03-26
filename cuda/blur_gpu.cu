#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

__global__ void blurKernel(unsigned char* input, unsigned char* output, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && y >= 1 && x < w-1 && y < h-1) {
        int sum = 0;
        for(int dx=-1; dx<=1; dx++)
            for(int dy=-1; dy<=1; dy++)
                sum += input[(y+dy)*w + (x+dx)];

        output[y*w + x] = sum / 9;
    }
}

int main() {
    Mat img = imread("../images/input.jpg", IMREAD_GRAYSCALE);

    int w = img.cols, h = img.rows;
    size_t size = w*h;

    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, img.data, size, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((w+15)/16, (h+15)/16);

    blurKernel<<<blocks, threads>>>(d_in, d_out, w, h);

    cudaMemcpy(img.data, d_out, size, cudaMemcpyDeviceToHost);

    imwrite("../outputs/cuda.jpg", img);

    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "CUDA Done\n";
    return 0;
}