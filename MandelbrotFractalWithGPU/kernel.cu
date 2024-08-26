﻿
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



#include <cmath> // for pow function

#include <iomanip> // For std::setprecision

#include "Remappers.h"

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
    double Ymin, double Ymax, double Xmin, double Xmax,
    int iter, int thresh,
    float degree, 
    cv::Mat* fractal_image,
    bool isVideo
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
    float k = 1.0;
    int M = floor(1080 * k), N = floor(1080 * k);
    double ratio = (double)N / (double)M;

    double Ymin = -1.7, Ymax = 1.7;
    double Xmin = -2.4;
    double Xmax = Xmin + (Ymax - Ymin) * ratio;

    int iter = 100;
    
    float thresh = 2;

    float degree = 2;
    

    /*
    double Ymin = 0.6859665904057353, Ymax = 0.9442179710300161;
    double Xmin = -0.3671721196466123;
    double Xmax = Xmin + (Ymax - Ymin) * ratio;

    int iter = 200;

    float thresh = 2;

    float degree = 2;
    */


    /*
    Size:
        Width: 918
        Height: 918

    Y coordinates:
            Y min: 0.6859665904057353
            Y max: 0.9442179710300161

    X coordinates:
            X min: -0.3671721196466123
            X max: -0.1089207390223315

    Other parameters:
            Iterations: 200
            Threshold: 2
            Degree: 3
    */



    bool isVideo;
    bool repeatMainMenu = true;

    int choice;
    while (repeatMainMenu) 
    {
        cout << "MENU:\n";
        cout << "1: Render image\n";
        cout << "2: Render video\n";
        cout << "Select an option: ";

        cin >> choice;

        switch (choice) 
        {
            case 1:
            {
                isVideo = false;
                repeatMainMenu = false;
                break;
            }
            case 2:
            {
                isVideo = true;
                repeatMainMenu = false;
                break;
            }
            default:
            {
                cout << "Not a valid option. Try again.\n\n";
                break;
            }
        }
    }

    /*
    //Test parameters
    M = 1080; M *= 0.9;
    N = M;


    Ymin =  0.00537518315;
    Ymax =  0.00537518415;
    Xmin = -1.7763135790;
    
    ratio = (double)N / (double)M;

    Xmax = Xmin + (Ymax - Ymin) * ratio;

    iter = 35000; thresh = 2;

    PARAMETERS:

    Size:
            Width: 972
            Height: 972

    Y coordinates:
            Y min: 0.003582497250219543
            Y max: 0.003582498269241166

    X coordinates:
            X min: -1.477394606418177
            X max: -1.477394605399155

    Other parameters:
            Iterations: 32500
            Threshold: 2
            Degree: 2



    Y coordinates:
        Y min: 0.6501447249932832
        Y max: 0.6501447264294704

X coordinates:
        X min: -0.1690813995583467
        X max: -0.1690813981221595

Other parameters:
        Iterations: 100000
        Threshold: 2
        Degree: 2



        Size:
        Width: 864
        Height: 864

        Y coordinates:
                Y min: 0.6550759647997856
                Y max: 0.6551523281769799

        X coordinates:
                X min: -0.1609347246595099
                X max: -0.1608583612823156

        Other parameters:
                Iterations: 4000
                Threshold: 2
                Degree: 2
    */

    
    cv::Mat fractal_image;

    int FrameStart = 0, FrameEnd = 1680; //56 seconds at 30fps
    double YminStart = Ymin, YminEnd = 0.6501447249932832;
    double YmaxStart = Ymax, YmaxEnd = 0.6501447264294704;
    double Y_limit = (YminEnd + YmaxEnd) / 2;

    double XminStart = Xmin, XminEnd = -0.1690813995583467;
    double XmaxEnd = XminEnd + (YmaxEnd - YminEnd) * ratio;
    double X_limit = (XminEnd + XmaxEnd) / 2;

    int iterStart = iter, iterEnd = 100000;
    float threshStart = thresh, threshEnd = 2;
    float degreeStart = degree, degreeEnd = 2;


    cudaError_t cudaStatus;

    if (!isVideo)
    {
        bool exploringFractal = true;

        float zoomFactor = 6.0f;
        float shiftFactor = 6.0f;

        while (exploringFractal)
        {
            // Recording the timestamp at the start of the code
            auto beg = high_resolution_clock::now();

            cudaStatus = mandelbrotWithCuda(N, M, Ymin, Ymax, Xmin, Xmax, iter, thresh, degree, &fractal_image, isVideo);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "mandelbrotWithCuda failed!");
                return 1;
            }
            cv::imshow("salida", fractal_image);

            // Taking a timestamp after the code is ran
            auto end = high_resolution_clock::now();

            auto duration = duration_cast<milliseconds>(end - beg);

            system("cls");

            // Displaying the elapsed time
            cout << "\nElapsed Time: " << duration.count() << " miliseconds.\n";

            cout << "\nPARAMETERS:\n";

            cout << "\nSize:\n";
            cout << "\tWidth: " << N << endl;
            cout << "\tHeight: " << M << endl;
            cout << "\nY coordinates:\n";
            cout << "\tY min: " << setprecision(numeric_limits<double>::digits10 + 1) << Ymin << endl;
            cout << "\tY max: " << setprecision(numeric_limits<double>::digits10 + 1) << Ymax << endl;
            cout << "\nX coordinates:\n";
            cout << "\tX min: " << setprecision(numeric_limits<double>::digits10 + 1) << Xmin << endl;
            cout << "\tX max: " << setprecision(numeric_limits<double>::digits10 + 1) << Xmax << endl;
            cout << "\nOther parameters:\n";
            cout << "\tIterations: " << setprecision(numeric_limits<double>::digits10 + 1) << iter << endl;
            cout << "\tThreshold: " << setprecision(numeric_limits<double>::digits10 + 1) << thresh << endl;
            cout << "\tDegree: " << setprecision(numeric_limits<double>::digits10 + 1) << degree << endl;
            
            bool repeatExplorerMenu = true;
            while (repeatExplorerMenu)
            {
                cout << "\nEXPLORER MENU:" << endl;
                cout << "\n1: Zoom in" << endl;
                cout << "2: Zoom out" << endl;
                cout << "3: Modify zoom factor (current: " << zoomFactor << ")" << endl;
                cout << "\n4: Shift up" << endl;
                cout << "5: Shift down" << endl;
                cout << "6: Shift left" << endl;
                cout << "7: Shift right" << endl;
                cout << "8: Modify shift factor (current: " << shiftFactor << ")" << endl;
                cout << "\n9: Increase/decrease degree" << endl;
                cout << "10: Increase/decrease threshold" << endl;
                cout << "11: Increase/decrease iterations" << endl;
                cout << "12: Display again" << endl;
                cout << "\n0: Exit" << endl;
                cout << "\n\tSelect an option (close image window first): ";
                cv::waitKey(0);

                cin >> choice;

                repeatExplorerMenu = false;

                switch (choice) 
                {
                    case 0: // Exit
                    {
                        exploringFractal = false;
                        break;
                    }
                    case 1: // Zoom in
                    {
                        double deltaY = Ymax - Ymin;
                        Ymin += deltaY / zoomFactor;
                        Ymax -= deltaY / zoomFactor;

                        double deltaX = Xmax - Xmin;
                        Xmin += deltaX / zoomFactor;
                        Xmax = Xmin + (Ymax - Ymin) * ratio;
                        break;
                    }
                    case 2: // Zoom out
                    {
                        double deltaY = Ymax - Ymin;
                        Ymin -= deltaY / (zoomFactor - 2);
                        Ymax += deltaY / (zoomFactor - 2);

                        double deltaX = Xmax - Xmin;
                        Xmin -= deltaX / (zoomFactor - 2);
                        Xmax = Xmin + (Ymax - Ymin) * ratio;
                        break;
                    }
                    case 3: // Modify zoom factor
                    {
                        cout << "\nChoose a zoom factor." << endl;
                        cout << "It's recommended to pick a number between 3 (strong) and 10 (weak)." << endl;
                        cout << "Current zoom factor: " << zoomFactor << "." << endl;
                        cout << "New zoom factor: ";
                        cin >> zoomFactor;
                        repeatExplorerMenu = true;
                        break;
                    }
                    case 4: // Shift up
                    {
                        double deltaY = Ymax - Ymin;
                        Ymin += deltaY / shiftFactor;
                        Ymax += deltaY / shiftFactor;
                        break;
                    }
                    case 5: // Shift down
                    {
                        double deltaY = Ymax - Ymin;
                        Ymin -= deltaY / shiftFactor;
                        Ymax -= deltaY / shiftFactor;
                        break;
                    }
                    case 6: // Shift left
                    {
                        double deltaX = Xmax - Xmin;
                        Xmin -= deltaX / shiftFactor;
                        Xmax -= deltaX / shiftFactor;
                        break;
                    }
                    case 7: // Shift right
                    {
                        double deltaX = Xmax - Xmin;
                        Xmin += deltaX / shiftFactor;
                        Xmax += deltaX / shiftFactor;
                        break;
                    }
                    case 8: // Modify shift factor
                    {
                        cout << "\nChoose a shift factor." << endl;
                        cout << "It's recommended to pick a number between 3 (strong) and 10 (weak)." << endl;
                        cout << "Current shift factor: " << shiftFactor << "." << endl;
                        cout << "New shift factor: ";
                        cin >> shiftFactor;
                        repeatExplorerMenu = true;
                        break;
                    }
                    case 9: // Increment/decrement degree
                    {
                        float degreeIncrement;
                        cout << "\nChoose an increment or decrement for the degree." << endl;
                        cout << "Use positive numbers for increments, negative for decrements." << endl;
                        cout << "Current degree: " << degree << "." << endl;
                        cout << "Increment by: ";
                        cin >> degreeIncrement;
                        degree += degreeIncrement;
                        break;
                    }
                    case 10: // Increment/decrement threshold
                    {
                        float threshIncrement;
                        cout << "\nChoose an increment or decrement for the threshold." << endl;
                        cout << "Use positive numbers for increments, negative for decrements." << endl;
                        cout << "Current threshold: " << thresh << "." << endl;
                        cout << "Increment by: ";
                        cin >> threshIncrement;
                        thresh += threshIncrement;
                        break;
                    }
                    case 11: // Increment/decrement threshold
                    {
                        int iterIncrement;
                        cout << "\nChoose an increment or decrement for the iterations." << endl;
                        cout << "Use positive integer numbers for increments, negative for decrements." << endl;
                        cout << "Current iterations: " << iter << "." << endl;
                        cout << "Increment by: ";
                        cin >> iterIncrement;
                        iter += iterIncrement;
                        break;
                    }
                    case 12: // Display again
                    {
                        break;
                    }
                    default:
                    {
                        repeatExplorerMenu = true;
                        cout << "Not a valid option. Try again.\n\n";
                        break;
                    }
                }

            }
        }
    }
    else
    {
        cout << "\nCREATING VIDEO\n\n";
        for (int index = FrameStart; index <= FrameEnd; index++)
        {
            Ymin = exponentialRemap(index, FrameEnd, YminStart, YminEnd, Y_limit);
            Ymax = exponentialRemap(index, FrameEnd, YmaxStart, YmaxEnd, Y_limit);

            Xmin = exponentialRemap(index, FrameEnd, XminStart, XminEnd, X_limit);
            Xmax = Xmin + (Ymax - Ymin) * ratio;

            //iter = exponentialRemap(index, FrameEnd, iterStart, iterEnd, 0);
            iter = expolinearRemap(index, 0, FrameEnd, iterStart, iterEnd, 0, 0.1); //0.1 en video5
            //iter = linearRemap(index, 0, FrameEnd, iterStart, iterEnd);
            thresh = exponentialRemap(index, FrameEnd, threshStart, threshEnd, 0);

            degree = linearRemap(index, 0, FrameEnd, degreeStart, degreeEnd);

            cudaStatus = mandelbrotWithCuda(N, M, Ymin, Ymax, Xmin, Xmax, iter, thresh, degree, &fractal_image, isVideo);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "mandelbrotWithCuda failed!");
                return 1;
            }

            std::ostringstream ss;
            ss << "D:/Git/MandelbrotFractalWithGPU/VideoFrames/Video8/" 
                << "Frame_" << index << ".png";
            std::string filename = ss.str();
            bool result = cv::imwrite(filename, fractal_image);
            if (result)
                cout << "Image " << index << " with " << iter << " iterations saved..." << endl;
            else
            {
                cerr << "\nError while saving image." << endl; 
                return 0;
            }
        }
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
    double Ymin, double Ymax, double Xmin, double Xmax,
    int iter, int thresh,
    float degree,
    cv::Mat* fractal_image,
    bool isVideo
)
{
    double Yaux = Ymin; //This mirrors the image in the y axis
    Ymin = -Ymax;
    Ymax = -Yaux;

    int pixelSize = width * height;

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
        if (!isVideo)
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
        if (!isVideo)
            fprintf(stderr, "\nSUCCESS in cudaDeviceSynchronize\n\n");
    }


    //openCV image. Use CV_8U if it's in grayscale
    cv::Mat frame = cv::Mat(cv::Size(width, height), CV_8UC3); 

    //Copy from GPU to CPU, over the image
    cudaMemcpy(frame.data, img_dev, pixelSize * sizeof(uchar) * 3, cudaMemcpyDeviceToHost);
    //cudaMemcpy(frame.ptr(), img_dev, pixelSize * sizeof(uchar) * 3, cudaMemcpyDeviceToHost); //alternative

    cv::applyColorMap(frame, *fractal_image, cv::COLORMAP_HOT);

    return cudaStatus;
}
