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
// gcc -n 32 ./q2 1000000 100

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
#define PARALLELIZABLE_FRACTION 0.95

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
 * Calculate the partial Euclidean distance between two vectors
 * @param p: The first vector
 * @param q: The second vector
 * @param n: The number of dimensions
 * @return: The partial Euclidean distance
 */
double euclidean_distance(int* p, int* q, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return sum;
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
}