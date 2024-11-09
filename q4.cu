/**
 * Author: Jason Gardner
 * Date: 11/6/2024
 * Class: COP6616 Parallel Computing
 * Instructor: Scott Piersall
 * Assignment: Homework 3
 * Filename: q4.cu
 * Description: This program computes the Euclidean distance between two points in 3D space using CUDA.
 */

/**
 * (11 points) Write a matrix multiplication program that uses GPU/CUDA. Your matrices should consist
 * of floating-point values (NOT INTEGERS). You may do this assignment in C, Python, and are free to create
 * a Jupyter notebook. If you do a Jupyter notebook, please include the notebook in your assignment report. 
 * You should run MANY experiments with varying sized matrices. What is the speedup? Is the speedup affected 
 * by matrix size? Explain (diagrams are useful here) how the GPU implementation of this code differs from 
 * a serial (CPU-based) implementation. 
 */

// nvcc -lm q4.cu -o q4
// ./q4 1000000 1000000 100

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#define MIN 0
#define MAX 99
#define TOLERANCE 0.000001
#define PARALLELIZABLE_FRACTION_DOUBLE 0.90
#define PARALLELIZABLE_FRACTION_INTEGER 0.80
#define BLOCK_SIZE 16
#define MAX_ELEMENTS 1000000000 // Threshold for warning about potential segmentation faults

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
double amdahl_speedup(int p, double PARALLELIZABLE_FRACTION) {
    return 1.0 / ((1.0 - PARALLELIZABLE_FRACTION) + (PARALLELIZABLE_FRACTION / p));
}

/**
 * Generate a random double value between MIN and MAX
 * @return: A random double value
 */
double random_double() {
    return MIN + ((double)rand() / RAND_MAX) * (MAX - MIN);
}

/**
 * Generate a random integer value between MIN and MAX
 * @return: A random integer value
 */
int random_int() {
    return MIN + rand() % (MAX - MIN + 1);
}

/** Function to compare two matrices with a tolerance */
bool compare_matrices(double* C_serial, double* C_parallel, int m, int k) {
    for (int i = 0; i < m * k; i++) {
        if (fabs(C_serial[i] - C_parallel[i]) > TOLERANCE) {
            return false;
        }
    }
    return true;
}

/** Matrix multiplication in serial (double) */
void matrixMultiplySerial(double* A, double* B, double* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            double sum = 0;
            for (int l = 0; l < n; l++) {
                sum += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

/** Matrix multiplication in serial (int) */
void matrixMultiplySerialInt(int* A, int* B, int* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            int sum = 0;
            for (int l = 0; l < n; l++) {
                sum += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

/** CUDA kernel for matrix multiplication using shared memory (double) */
__global__ void matrixMultiplyKernelShared(double* A, double* B, double* C, int m, int n, int k) {
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;

    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        if (row < m && (t * BLOCK_SIZE + threadIdx.x) < n) {
            As[threadIdx.y][threadIdx.x] = A[row * n + t * BLOCK_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if ((t * BLOCK_SIZE + threadIdx.y) < n && col < k) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * k + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            value += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < k) {
        C[row * k + col] = value;
    }
}

/** CUDA kernel for matrix multiplication (int) */
__global__ void matrixMultiplyKernelInt(int* A, int* B, int* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        int value = 0;
        for (int i = 0; i < n; i++) {
            value += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = value;
    }
}

/** Host function for matrix multiplication using CUDA (double) */
void matrixMultiplyCUDA(double* A, double* B, double* C, int m, int n, int k) {
    double *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc((void**)&d_A, m * n * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for A: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&d_B, n * k * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for B: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&d_C, m * k * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for C: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixMultiplyKernelShared<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(C, d_C, m * k * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/** Host function for matrix multiplication using CUDA (int) */
void matrixMultiplyCUDAInt(int* A, int* B, int* C, int m, int n, int k) {
    int *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc((void**)&d_A, m * n * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for A: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&d_B, n * k * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for B: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&d_C, m * k * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for C: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixMultiplyKernelInt<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(C, d_C, m * k * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/** Main function */
int main(int argc, char** argv) {
    // Command line argument variables
    unsigned int m = 512, n = 512, k = 512, num_runs = 10;

    // Parse command line arguments or use default values
    if (argc > 1) {
        if (sscanf(argv[1], "%u", &m) != 1) {
            printf("Invalid value for m. Using default: m = 512\n");
            m = 512;
        }
    } else {
        printf("Missing value for m. Using default: m = 512\n");
    }

    if (argc > 2) {
        if (sscanf(argv[2], "%u", &n) != 1) {
            printf("Invalid value for n. Using default: n = 512\n");
            n = 512;
        }
    } else {
        printf("Missing value for n. Using default: n = 512\n");
    }

    if (argc > 3) {
        if (sscanf(argv[3], "%u", &k) != 1) {
            printf("Invalid value for k. Using default: k = 512\n");
            k = 512;
        }
    } else {
        printf("Missing value for k. Using default: k = 512\n");
    }

    if (argc > 4) {
        if (sscanf(argv[4], "%u", &num_runs) != 1) {
            printf("Invalid value for num_runs. Using default: num_runs = 10\n");
            num_runs = 10;
        }
    } else {
        printf("Missing value for num_runs. Using default: num_runs = 10\n");
    }

    // Validate command line arguments
    if (m < 1 || n < 1 || k < 1 || num_runs < 1) {
        fprintf(stderr, "Matrix dimensions and number of runs must be greater than 0!\n");
        exit(EXIT_FAILURE);
    }

    // Check if the total number of elements is too large
    unsigned long long total_elements = (unsigned long long)m * n * k;
    if (total_elements > MAX_ELEMENTS) {
        unsigned long long estimated_memory = (unsigned long long)(m * n * sizeof(double) + n * k * sizeof(double) + m * k * sizeof(double));
        double estimated_memory_mb = estimated_memory / (1024.0 * 1024.0);
        double estimated_memory_gb = estimated_memory / (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr, "Warning: The total number of elements (%llu) may be too large and could lead to a segmentation fault due to", total_elements);
        fprintf(stderr, "\ninsufficient memory! Estimated memory usage: %.2lf MB (%.2lf GB).\n", estimated_memory_mb, estimated_memory_gb);
    }

    // Allocate memory for matrices (double)
    double* A_d = (double*)malloc(m * n * sizeof(double));
    double* B_d = (double*)malloc(n * k * sizeof(double));
    double* C_serial_d = (double*)malloc(m * k * sizeof(double));
    double* C_parallel_d = (double*)malloc(m * k * sizeof(double));

    if (!A_d || !B_d || !C_serial_d || !C_parallel_d) {
        fprintf(stderr, "Matrix memory allocation failed for double matrices!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for matrices (int)
    int* A_i = (int*)malloc(m * n * sizeof(int));
    int* B_i = (int*)malloc(n * k * sizeof(int));
    int* C_serial_i = (int*)malloc(m * k * sizeof(int));
    int* C_parallel_i = (int*)malloc(m * k * sizeof(int));

    if (!A_i || !B_i || !C_serial_i || !C_parallel_i) {
        fprintf(stderr, "Matrix memory allocation failed for int matrices!\n");
        exit(EXIT_FAILURE);
    }

    // Fill matrices with random values
    seed_random();
    for (int i = 0; i < m * n; i++) {
        A_d[i] = random_double();
        A_i[i] = random_int();
    }
    for (int i = 0; i < n * k; i++) {
        B_d[i] = random_double();
        B_i[i] = random_int();
    }

    // Variables for tracking time
    double total_parallel_time_d = 0;
    double total_serial_time_d = 0;
    double total_parallel_time_i = 0;
    double total_serial_time_i = 0;

    // Perform the matrix multiplication for num_runs (double)
    for (unsigned int run = 0; run < num_runs; run++) {
        Stopwatch parallel_timer, serial_timer;

        // Start parallel timing (double)
        clock_gettime(CLOCK_MONOTONIC, &parallel_timer.start);

        // Perform parallel matrix multiplication (double)
        matrixMultiplyCUDA(A_d, B_d, C_parallel_d, m, n, k);

        // Stop parallel timing (double)
        clock_gettime(CLOCK_MONOTONIC, &parallel_timer.stop);
        total_parallel_time_d += calculate_time(parallel_timer);

        // Start serial timing (double)
        clock_gettime(CLOCK_MONOTONIC, &serial_timer.start);

        // Perform serial matrix multiplication (double)
        matrixMultiplySerial(A_d, B_d, C_serial_d, m, n, k);

        clock_gettime(CLOCK_MONOTONIC, &serial_timer.stop);
        total_serial_time_d += calculate_time(serial_timer);

        // Compare results (double)
        if (!compare_matrices(C_serial_d, C_parallel_d, m, k)) {
            fprintf(stderr, "Error: Double matrices do not match in run %u!\n", run + 1);
        }
    }

    // Perform the matrix multiplication for num_runs (int)
    for (unsigned int run = 0; run < num_runs; run++) {
        Stopwatch parallel_timer, serial_timer;

        // Start parallel timing (int)
        clock_gettime(CLOCK_MONOTONIC, &parallel_timer.start);

        // Perform parallel matrix multiplication (int)
        matrixMultiplyCUDAInt(A_i, B_i, C_parallel_i, m, n, k);

        // Stop parallel timing (int)
        clock_gettime(CLOCK_MONOTONIC, &parallel_timer.stop);
        total_parallel_time_i += calculate_time(parallel_timer);

        // Start serial timing (int)
        clock_gettime(CLOCK_MONOTONIC, &serial_timer.start);

        // Perform serial matrix multiplication (int)
        matrixMultiplySerialInt(A_i, B_i, C_serial_i, m, n, k);

        clock_gettime(CLOCK_MONOTONIC, &serial_timer.stop);
        total_serial_time_i += calculate_time(serial_timer);

        // Compare results (int)
        if (memcmp(C_serial_i, C_parallel_i, m * k * sizeof(int)) != 0) {
            fprintf(stderr, "Error: Integer matrices do not match in run %u!\n", run + 1);
        }
    }

    // Calculate average times
    double average_parallel_time_d = total_parallel_time_d / num_runs;
    double average_serial_time_d = total_serial_time_d / num_runs;
    double average_parallel_time_i = total_parallel_time_i / num_runs;
    double average_serial_time_i = total_serial_time_i / num_runs;

    // Output results and speedup calculations (double)
    double actual_speedup_d = average_serial_time_d / average_parallel_time_d;
    double theoretical_speedup_d = amdahl_speedup(m, PARALLELIZABLE_FRACTION_DOUBLE);
    double speedup_ratio_d = (actual_speedup_d / theoretical_speedup_d) * 100;

    printf("Double Matrix Multiplication:\n");
    printf("Average Serial Time:\t\t\t%lfs\n", average_serial_time_d);
    printf("Average Parallel Time:\t\t\t%lfs\n", average_parallel_time_d);
    printf("Theoretical Speedup [Amdahl's Law]:\t%lf\n", theoretical_speedup_d);
    printf("Actual Speedup:\t\t\t\t%lf\n", actual_speedup_d);
    printf("Speedup Efficiency:\t\t\t%lf%%\n", speedup_ratio_d);

    // Output results and speedup calculations (int)
    double actual_speedup_i = average_serial_time_i / average_parallel_time_i;
    double theoretical_speedup_i = amdahl_speedup(m, PARALLELIZABLE_FRACTION_INTEGER);
    double speedup_ratio_i = (actual_speedup_i / theoretical_speedup_i) * 100;

    printf("\nInteger Matrix Multiplication:\n");
    printf("Average Serial Time:\t\t\t%lfs\n", average_serial_time_i);
    printf("Average Parallel Time:\t\t\t%lfs\n", average_parallel_time_i);
    printf("Theoretical Speedup [Amdahl's Law]:\t%lf\n", theoretical_speedup_i);
    printf("Actual Speedup:\t\t\t\t%lf\n", actual_speedup_i);
    printf("Speedup Efficiency:\t\t\t%lf%%\n", speedup_ratio_i);

    printf("\nDifference in Average Serial Time (Double vs Integer):\t\t%lfs\n", average_serial_time_d - average_serial_time_i);
    printf("Difference in Average Parallel Time (Double vs Integer):\t%lfs\n", average_parallel_time_d - average_parallel_time_i);
    printf("Difference in Theoretical Speedup (Double vs Integer):\t\t%lf\n", theoretical_speedup_d - theoretical_speedup_i);
    printf("Difference in Actual Speedup (Double vs Integer):\t\t%lf\n", actual_speedup_d - actual_speedup_i);
    printf("Difference in Speedup Efficiency (Double vs Integer):\t\t%lf%%\n", speedup_ratio_d - speedup_ratio_i);


    // Free allocated memory
    free(A_d);
    free(B_d);
    free(C_serial_d);
    free(C_parallel_d);
    free(A_i);
    free(B_i);
    free(C_serial_i);
    free(C_parallel_i);

    return 0;
}