
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <opencv2/highgui/highgui.hpp> //To use uchar
/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>

#include <iostream>
*/

cudaError_t addWithCuda(int width, int height);

__global__ void myKernel(int width, int height, uchar* image)
{
	//columna
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	//fila 
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < width && j < height) //estamos dentro de la imagen
	{
		int idx = (j * width + i) * 3;

        //los floats se convierten en uchar de forma implicita.
        image[idx] = 255; //Blue
        image[idx + 1] = 0; //Green
        image[idx + 2] = 128; //Red
	}
}

int main()
{
    int width = 1000, height = 500;
    int pixelSize = width * height;

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(width, height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
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
cudaError_t addWithCuda(int width, int height)
{
    int pixelSize = width * height;

    uchar* img_dev;

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

    // Launch a kernel on the GPU with one thread for each element.
    myKernel <<<blocks, threads>>>(width, height, img_dev);

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

    //openCV image. Use CV_8U if it's in grayscale
    cv::Mat frame = cv::Mat(cv::Size(width, height), CV_8UC3); 

    //Copy from GPU to CPU, over the image
    cudaMemcpy(frame.data, img_dev, pixelSize * sizeof(uchar) * 3, cudaMemcpyDeviceToHost);
    //cudaMemcpy(frame.ptr(), img_dev, pixelSize * sizeof(uchar) * 3, cudaMemcpyDeviceToHost); //alternative

    cv::imshow("salida", frame);

    cv::waitKey(0);

    return cudaStatus;
}
