
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <complex>

#include <opencv2/highgui/highgui.hpp> //To use uchar


#include <thrust/complex.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <iostream>

using namespace std::chrono;

using namespace std;
using cmplxDouble = complex<double>;
/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>

#include <iostream>
*/


cudaError_t mandelbrotWithCuda
(
    int width, int height,
    double Ymin, double Ymax, double Xmin,
    int iter, int thresh,
    float degree
);

__device__ uchar MandelbrotIteration(thrust::complex<double> C, int iterations, float thresh, float degree)
{
    uchar greyLevel;
    thrust::complex<double> Zn = C;

    for (int i = 1; i <= iterations; i++)
    {
        Zn = Zn;
        Zn = pow(Zn, degree) + C;
        double magnitude_squared = norm(Zn); // squared magnitude of Zn
        if (magnitude_squared > thresh * thresh)
        {
            greyLevel = static_cast<uchar>(floor(255.0 - (255.0 * (i - 1) / iterations)));
            return greyLevel;
        }
        /*
        */
    }
    return 0;
}

__global__ void myKernel
(
    uchar* image, 
    int width, int height, 
    double deltaX, double deltaY,
    double Xmin, double Ymin,
    int iter, float thresh, float degree
)
{
	//columna
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	//fila 
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < width && j < height) //estamos dentro de la imagen
	{
		int idx = (j * width + i) * 3;

        double real = Xmin + deltaX * i; //real number
        double imag = Ymin + deltaY * j; //imaginary number
        
        thrust::complex<double> z(real, imag);

        uchar grey = MandelbrotIteration(z, iter, thresh, degree);

        grey = 255 - grey;

        if (grey == 255)
            grey = 0;

        //uchar grey = i;

        //los floats se convierten en uchar de forma implicita.


        image[idx] = grey; //Blue
        image[idx + 1] = grey; //Green
        image[idx + 2] = grey; //Red
	}
}

int main()
{
    int k = 1;
    int M = floor(513 * k), N = floor(1024 * k);

    double Ymin = -1.1f, Ymax = 1.1f;
    double Xmin = -2.6; //Xmax is calculated with the ratio N/M

    int iter = 100;
    
    float thresh = 2;

    float degree = 2;



    //Test parameters
    M = 1080; M *= 0.9;
    N = M;

    Ymin =  0.00050;
    Ymax =  0.00600;
    Xmin = -1.77890;

    iter = 300; thresh = 2;



    // Add vectors in parallel.
    cudaError_t cudaStatus = mandelbrotWithCuda(N, M, Ymin, Ymax, Xmin, iter, thresh, degree);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mandelbrotWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t mandelbrotWithCuda(
    int width, int height,
    double Ymin, double Ymax, double Xmin,
    int iter, int thresh,
    float degree
)
{
    double Yaux = Ymin; //This mirrors the image in the y axis
    Ymin = -Ymax;
    Ymax = -Yaux;

    int pixelSize = width * height;
    
    double ratio = (double)width / (double)height;

    double Xmax = Xmin + (Ymax - Ymin) * ratio;

    uchar* img_dev;

    double deltaX = (Xmax - Xmin) / (double)width;
    double deltaY = (Ymax - Ymin) / (double)height;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Allocate memory
    cudaStatus = cudaMalloc(&img_dev, pixelSize * sizeof(uchar) * 3); //uchar and char weight 1 byte, so this multiplication is unnecessary
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    /*
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    */

    dim3 threads(16, 16); // = 256 pixels
    dim3 blocks(ceil((float)width / (float)threads.x), ceil((float)height / (float)threads.y));

    // Recording the timestamp at the start of the code
    auto beg = high_resolution_clock::now();

    // Launch a kernel on the GPU with one thread for each element.
    myKernel <<<blocks, threads>>>
    (
        img_dev, width, height, deltaX, deltaY, Xmin, Ymin, iter, thresh, degree
    );

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "myKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    else
    {
        fprintf(stderr, "\nSUCCESS in cudaGetLastError\n");
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }
    else
    {
        fprintf(stderr, "\nSUCCESS in cudaDeviceSynchronize\n\n");
    }

    // Taking a timestamp after the code is ran
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - beg);

    // Displaying the elapsed time
    std::cout << "\nElapsed Time: " << duration.count() << " miliseconds.\n\n\n";


    //openCV image. Use CV_8U if it's in grayscale
    cv::Mat frame = cv::Mat(cv::Size(width, height), CV_8UC3); 

    //Copy from GPU to CPU, over the image
    cudaMemcpy(frame.data, img_dev, pixelSize * sizeof(uchar) * 3, cudaMemcpyDeviceToHost);
    //cudaMemcpy(frame.ptr(), img_dev, pixelSize * sizeof(uchar) * 3, cudaMemcpyDeviceToHost); //alternative

    cv::Mat img_color;

    cv::applyColorMap(frame, img_color, cv::COLORMAP_HOT);

    cv::imshow("salida", img_color);

    cv::waitKey(0);

    return cudaStatus;
}
