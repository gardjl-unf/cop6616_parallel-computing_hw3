/**
 * Author: Jason Gardner
 * Date: 11/6/2024
 * Class: COP6616 Parallel Computing
 * Instructor: Scott Piersall
 * Assignment: Homework 3
 * Filename: q2.c
 * Description: This program computes the Euclidean distance between two points in 3D space using CUDA.
 */

/**
 * (15 points)  Please write a CUDA program to compute the Euclidean distance, similar to the problem in assignment 2.
 * You may design your code using the following steps:
 *      Declare the arrays (host and device). All arrays should be dynamically allocated; the host arrays can be allocated either 
 *          with malloc or new, while the device arrays should be allocated with cudaMalloc.
 *      Print the number of CUDA-enabled hardware devices attached to the system by calling cudaGetDeviceCount.
 *      Print at least 3 interesting properties of Device 0, including the device name, by calling cudaGetDeviceProperties. 
 *          The first argument to this function is a pointer to a struct of type cudaDeviceProp. 
 *      Calls InitArray to initialize the host arrays. InitArray initializes an integer array with random numbers within a 
 *          fairly small range (0 to 99).
 *      Calls cudaMemcpy to copy the host input arrays to the device.
 *      Calls the CUDA kernel, which computes the square of the difference of the components for each dimension, reduce all the 
 *          elements of the output array in parallel (you may need to investigate how to implement an efficient parallel reduce in CUDA), 
 *          and takes the square root of the sum.
 *      Run experiments using varying size of inputs. Graph and discuss the speedup provided by the GPU/CUDA implementation over 
 *          varying input sizes. Is there an input size where the speedup stops?
 */

// nvcc -lm q2.cu -o q2
// ./q2 1000000 100

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
// Interestingly, _Bool does not work in nvcc
#include <stdbool.h>
#include <math.h>

#define MIN 0
#define MAX 99
#define TOLERANCE 0.0001
#define PARALLELIZABLE_FRACTION 0.50
#define BLOCK_SIZE 1024

/** Struct to store start and stop times */
typedef struct {
    struct timespec start;
    struct timespec stop;
} Stopwatch;

/** Seed the random number generator with entropy from /dev/urandom */
void seed_random() {
    int fd = open("/dev/urandom", O_RDONLY);
    unsigned int seed;
    read(fd, &seed, sizeof(seed));
    close(fd);
    srandom(seed);
}

/** Calculate time in seconds
 * @param timer: The stopwatch struct
 * @return: The time in seconds
 */
double calculate_time(Stopwatch timer) {
    return (timer.stop.tv_sec - timer.start.tv_sec) + (timer.stop.tv_nsec - timer.start.tv_nsec) / 1e9;
}

/** Calculate the theoretical speedup using Amdahl's law */
double amdahl_speedup(int p) {
    return 1.0 / ((1.0 - PARALLELIZABLE_FRACTION) + (PARALLELIZABLE_FRACTION / p));
}

/**
 * Calculate the Euclidean distance between two vectors in serial
 * @param p: The first vector
 * @param q: The second vector
 * @param n: The number of dimensions
 * @return: The partial Euclidean distance
 */
double euclideanDistanceSerial(int* p, int* q, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return sqrt(sum);
}

/**
 * Calculate a random vector value between MIN and MAX
 * @return: A random vector value between MIN and MAX
 */
int random_vector_value() {
    return MIN + rand() % (MAX - MIN + 1);
}

/** Function to compare serial and parallel results with a tolerance */
bool compare_result(double result_s, double result_m) {
    return fabs(result_s - result_m) < TOLERANCE;
}

/** Display CUDA Information */
void display_cuda_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("Number of CUDA-enabled devices: %d\n", device_count);

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);
    printf("Device Name: %s\n", device_properties.name);
    printf("Compute Capability: %d.%d\n", device_properties.major, device_properties.minor);
    printf("Total Global Memory: %lu bytes\n", device_properties.totalGlobalMem);
    printf("Shared Memory Per Block: %lu bytes\n", device_properties.sharedMemPerBlock);
    printf("Registers Per Block: %d\n", device_properties.regsPerBlock);
}

// CUDA kernel to compute squared differences
__global__ void squaredDifferenceKernel(const int *a, const int *b, double *result, const int m) {
    // Calculate starting index for this thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Compute squared difference if within bounds
    if (i < m) {
        int diff = a[i] - b[i];
        result[i] = (double)(diff * diff);
    }
}

// Custom atomicAdd function for double because apparently I'm using some antiquated version of CUDA
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// CUDA kernel for parallel reduction within a block
__global__ void reduceKernel(double *input, double *output, int size) {
    __shared__ double sharedData[BLOCK_SIZE];
    
    int threadId = threadIdx.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Load elements into shared memory
    sharedData[threadId] = (i < size) ? input[i] : 0.0;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadId < stride) {
            sharedData[threadId] += sharedData[threadId + stride];
        }
        __syncthreads();
    }

    // Write result of this block's reduction to global memory
    if (threadId == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// CUDA kernel to perform final reduction across blocks
__global__ void finalReductionKernel(double *input, double *output, int size) {
    __shared__ double sharedData[BLOCK_SIZE];

    int threadId = threadIdx.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Load elements into shared memory
    sharedData[threadId] = (i < size) ? input[i] : 0.0;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadId < stride) {
            sharedData[threadId] += sharedData[threadId + stride];
        }
        __syncthreads();
    }

    // Write result of this block's reduction to global memory
    if (threadId == 0) {
        atomicAddDouble(output, sharedData[0]);
    }
}

// Host function to compute Euclidean distance
double euclideanDistanceCUDA(const int *a, const int *b, int m) {
    int *d_a, *d_b;
    double *d_temp, *d_block_sums, *d_result;

    // Allocate memory on the device for vectors and the temporary result array
    cudaMalloc((void**)&d_a, m * sizeof(int));
    cudaMalloc((void**)&d_b, m * sizeof(int));
    cudaMalloc((void**)&d_temp, m * sizeof(double));
    cudaMalloc((void**)&d_block_sums, ((m + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(double));
    cudaMalloc((void**)&d_result, sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_a, a, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, m * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize result to 0
    double initial_result = 0.0;
    cudaMemcpy(d_result, &initial_result, sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel to compute squared differences
    int numBlocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    squaredDifferenceKernel<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_temp, m);

    // Perform reduction to sum up squared differences within each block
    reduceKernel<<<numBlocks, BLOCK_SIZE>>>(d_temp, d_block_sums, m);

    // Launch final reduction kernel to sum up all block results
    int finalNumBlocks = (numBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    finalReductionKernel<<<finalNumBlocks, BLOCK_SIZE>>>(d_block_sums, d_result, numBlocks);

    // Copy the final result to host
    double sum;
    cudaMemcpy(&sum, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_temp);
    cudaFree(d_block_sums);
    cudaFree(d_result);

    // Return the square root of the sum to compute the Euclidean distance
    return sqrt(sum);
}

/**
 * Main function
 * @param argc: Number of arguments
 * @param argv: Array of arguments
 * @return: 0 if successful
 */
int main(int argc, char** argv) {
    // Command line argument variables
    unsigned int num_runs;
    unsigned int m;

    // Parse command line arguments or use default values
    if (argc < 3) {
        printf("Usage:\t\t\t%s <vector dimension> <number of runs to average>\n", argv[0]);
        if (argc < 3) {
            printf("Using default values:\tnum_runs = 100, m = 1000000\n\n");
            m = 1000000;   
            num_runs = 100;
        }
        else if (argc < 2) {
            printf("Using default value:\tnum_runs = 100\n\n");
            num_runs = 100;
            sscanf(argv[1], "%u", &m);
        }
        
    }
    else {
        sscanf(argv[1], "%u", &m);
        sscanf(argv[2], "%u", &num_runs);
    }

    // Validate command line arguments
    if (m < 1) {
        fprintf(stderr, "Vector dimension must be greater than 0!\n");
        exit(EXIT_FAILURE);
    }
    if (num_runs < 1) {
        fprintf(stderr, "Number of runs must be greater than 0!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate vector memory
    int* vector1 = (int*) malloc(m * sizeof(int));
    int* vector2 = (int*) malloc(m * sizeof(int));

    if (!vector1 || !vector2) {
        fprintf(stderr, "Vector memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    // Fill vectors with random values
    seed_random();
    for (int i = 0; i < m; i++) {
        vector1[i] = random_vector_value();
        vector2[i] = random_vector_value();
    }

    // Variables for tracking time
    double total_parallel_time = 0;
    double total_serial_time = 0;

    display_cuda_info();

    // Perform the Euclidean distance calculation for num_runs
    for (int run = 0; run < num_runs; run++) {
        Stopwatch parallel_timer, serial_timer;

        // Start parallel timing
        clock_gettime(CLOCK_MONOTONIC, &parallel_timer.start);

        // Perform parallel Euclidean distance calculation
        double result_c = euclideanDistanceCUDA(vector1, vector2, m);

        // Stop parallel timing
        clock_gettime(CLOCK_MONOTONIC, &parallel_timer.stop);
        total_parallel_time += calculate_time(parallel_timer);

        // Start serial timing
        clock_gettime(CLOCK_MONOTONIC, &serial_timer.start);

        // Perform serial Euclidean distance calculation
        double result_s = euclideanDistanceSerial(vector1, vector2, m);

        clock_gettime(CLOCK_MONOTONIC, &serial_timer.stop);
        total_serial_time += calculate_time(serial_timer);

        // Compare results after each run
        if (!compare_result(result_s, result_c)) {
            printf("Results do not match in run %d! Serial: %lf, Parallel: %lf\n", run + 1, result_s, result_c);
        }
    }

    // Calculate average times
    double average_parallel_time = total_parallel_time / num_runs;
    double average_serial_time = total_serial_time / num_runs;

    // Output results and speedup calculations
    double actual_speedup = average_serial_time / average_parallel_time;
    double theoretical_speedup = amdahl_speedup(m);
    double speedup_ratio = (actual_speedup / theoretical_speedup) * 100;

    printf("Average Serial Time:\t\t\t%lfs\n", average_serial_time);
    printf("Average Parallel Time:\t\t\t%lfs\n", average_parallel_time);
    printf("Theoretical Speedup [Amdahl's Law]:\t%lf\n", theoretical_speedup);
    printf("Actual Speedup:\t\t\t\t%lf\n", actual_speedup);
    printf("Speedup Efficiency:\t\t\t%lf%%\n", speedup_ratio);

    // Free allocated memory
    free(vector1);
    free(vector2);

    return 0;
}
