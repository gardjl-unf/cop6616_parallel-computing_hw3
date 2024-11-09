/**
 * Author: Jason Gardner
 * Date: 11/6/2024
 * Class: COP6616 Parallel Computing
 * Instructor: Scott Piersall
 * Assignment: Homework 3
 * Filename: q3.c
 * Description: This program computes the Euclidean distance between two points in 3D space using CUDA.
 * 
 * INPUT IMAGES ARE OWNED BY CATHERINE ANDERSON OF THE PAINTED ME, LLC AND ARE USED FOR EDUCATIONAL PURPOSES ONLY
 * SHE IS MY PARTNER AND I HAVE PERMISSION TO USE THEM (I AM ALSO A PART OWNER OF THE PAINTED ME, LLC)
 * IMAGE SOURCE: https://www.thepaintedme.com/
 */

/**
 * (10 points) You are working on team that is building an edge detection module. The first step in edge detection 
 * is to remove color information and work directly in black-and-white. You do not need to do edge detection, just 
 * the first step: removing the color information from images. So, please write a CUDA program to convert a color 
 * image to grayscale using the Colorimetric method. I suggest that you use PNG or BMP images as input.  I want 
 * to see the original images that you tested with and the output/resulting image. You may do this assignment in 
 * C or Python, and are free to create a Jupyter notebook. If you do a Jupyter notebook, please include the notebook 
 * in your assignment report. How does using a GPU for this vs. a serial CPU-based implementation perform? Is there 
 * a speedup?  How (be specific) is the GPU architecture well-suited to this task?
 */

// nvcc -lm -lpng q3.cu -o q3
// ./q3

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <png.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
// Interestingly, _Bool does not work in nvcc
#include <stdbool.h>
#include <math.h>

#define PARALLELIZABLE_FRACTION 0.50
#define BLOCK_SIZE 1024
#define NUM_IMAGES 7
#define INPUT_LOCATION "./in/"
#define OUTPUT_LOCATION "./out/"
#define RGB_VALUES 0.2126, 0.7152, 0.0722

/** Struct to store start and stop times */
typedef struct {
    struct timespec start;
    struct timespec stop;
} Stopwatch;

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

// Convert PNG image to greyscale and save the output (Serial version)
void convertPNGToGreyScaleSerial(char* filename, char* output_filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Error creating read struct\n");
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Error creating info struct\n");
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Error during init_io\n");
        exit(EXIT_FAILURE);
    }

    png_init_io(png, file);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_RGBA) png_set_rgb_to_gray(png, PNG_ERROR_ACTION_NONE, -1, -1);

    png_read_update_info(png, info);

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, row_pointers);
    fclose(file);

    // Write greyscale image to output file
    FILE* output_file = fopen(output_filename, "wb");
    if (!output_file) {
        fprintf(stderr, "Error opening output file %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    png_structp png_out = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_out) {
        fprintf(stderr, "Error creating write struct\n");
        exit(EXIT_FAILURE);
    }

    png_infop info_out = png_create_info_struct(png_out);
    if (!info_out) {
        fprintf(stderr, "Error creating info struct\n");
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png_out))) {
        fprintf(stderr, "Error during writing header\n");
        exit(EXIT_FAILURE);
    }

    png_init_io(png_out, output_file);
    png_set_IHDR(
        png_out,
        info_out,
        width, height,
        8,
        PNG_COLOR_TYPE_GRAY,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png_out, info_out);

    if (setjmp(png_jmpbuf(png_out))) {
        fprintf(stderr, "Error during writing bytes\n");
        exit(EXIT_FAILURE);
    }

    png_write_image(png_out, row_pointers);

    if (setjmp(png_jmpbuf(png_out))) {
        fprintf(stderr, "Error during end of write\n");
        exit(EXIT_FAILURE);
    }

    png_write_end(png_out, NULL);

    // Cleanup
    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);

    fclose(output_file);
}

// Convert PNG image to greyscale and save the output (CUDA version)
__global__ void greyscale_kernel(unsigned char* d_input, unsigned char* d_output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        int r = d_input[3 * idx];
        int g = d_input[3 * idx + 1];
        int b = d_input[3 * idx + 2];
        d_output[idx] = (unsigned char)(0.2126f * r + 0.7152f * g + 0.0722f * b);
    }
}

void convertPNGToGreyScaleCUDA(char* filename, char* output_filename) {
    // Load the image using libpng (similar to serial function)
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Error creating read struct\n");
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Error creating info struct\n");
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Error during init_io\n");
        exit(EXIT_FAILURE);
    }

    png_init_io(png, file);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_RGBA) png_set_rgb_to_gray(png, PNG_ERROR_ACTION_NONE, -1, -1);

    png_read_update_info(png, info);

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, row_pointers);
    fclose(file);

    // Copy image data to linear buffer
    unsigned char* h_input = (unsigned char*)malloc(3 * width * height);
    unsigned char* h_output = (unsigned char*)malloc(width * height);
    for (int y = 0; y < height; y++) {
        memcpy(h_input + y * width * 3, row_pointers[y], 3 * width);
    }

    // Allocate device memory and copy data to GPU
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, 3 * width * height);
    cudaMalloc(&d_output, width * height);
    cudaMemcpy(d_input, h_input, 3 * width * height, cudaMemcpyHostToDevice);

    // Launch kernel
    int num_threads = BLOCK_SIZE;
    int num_blocks = (width * height + num_threads - 1) / num_threads;
    greyscale_kernel<<<num_blocks, num_threads>>>(d_input, d_output, width, height);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, width * height, cudaMemcpyDeviceToHost);

    // Write greyscale image to output file
    FILE* output_file = fopen(output_filename, "wb");
    if (!output_file) {
        fprintf(stderr, "Error opening output file %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    png_structp png_out = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_out) {
        fprintf(stderr, "Error creating write struct\n");
        exit(EXIT_FAILURE);
    }

    png_infop info_out = png_create_info_struct(png_out);
    if (!info_out) {
        fprintf(stderr, "Error creating info struct\n");
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png_out))) {
        fprintf(stderr, "Error during writing header\n");
        exit(EXIT_FAILURE);
    }

    png_init_io(png_out, output_file);
    png_set_IHDR(
        png_out,
        info_out,
        width, height,
        8,
        PNG_COLOR_TYPE_GRAY,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png_out, info_out);

    if (setjmp(png_jmpbuf(png_out))) {
        fprintf(stderr, "Error during writing bytes\n");
        exit(EXIT_FAILURE);
    }

    for (int y = 0; y < height; y++) {
        png_write_row(png_out, h_output + y * width);
    }

    if (setjmp(png_jmpbuf(png_out))) {
        fprintf(stderr, "Error during end of write\n");
        exit(EXIT_FAILURE);
    }

    png_write_end(png_out, NULL);

    // Cleanup
    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    fclose(output_file);
}

/**
 * Main function
 * @param argc: Number of arguments
 * @param argv: Array of arguments
 * @return: 0 if successful
 */
int main(int argc, char** argv) {
    char input_filename[256];
    char output_filename_s[256];
    char output_filename_c[256];

    for (int image_num = 1; image_num < NUM_IMAGES; image_num++) {
        sprintf(input_filename, "%s%d.png", INPUT_LOCATION, image_num);
        sprintf(output_filename_s, "%simage_%d_s.png", OUTPUT_LOCATION, image_num);
        sprintf(output_filename_c, "%simage_%d_c.png", OUTPUT_LOCATION, image_num);

        Stopwatch parallel_timer, serial_timer;

        // Start serial timing
        clock_gettime(CLOCK_MONOTONIC, &serial_timer.start);
        convertPNGToGreyScaleSerial(input_filename, output_filename_s);
        clock_gettime(CLOCK_MONOTONIC, &serial_timer.stop);
        double total_serial_time = calculate_time(serial_timer);

        // Start parallel timing
        clock_gettime(CLOCK_MONOTONIC, &parallel_timer.start);
        convertPNGToGreyScaleCUDA(input_filename, output_filename_c);
        clock_gettime(CLOCK_MONOTONIC, &parallel_timer.stop);
        double total_parallel_time = calculate_time(parallel_timer);

        // Output results and speedup calculations
        double actual_speedup = total_serial_time / total_parallel_time;
        double theoretical_speedup = amdahl_speedup(BLOCK_SIZE);
        double speedup_ratio = (actual_speedup / theoretical_speedup) * 100;

        printf("Image %d:\n", image_num);
        printf("Average Serial Time:\t\t\t%lfs\n", total_serial_time);
        printf("Average Parallel Time:\t\t\t%lfs\n", total_parallel_time);
        printf("Theoretical Speedup [Amdahl's Law]:\t%lf\n", theoretical_speedup);
        printf("Actual Speedup:\t\t\t%lf\n", actual_speedup);
        printf("Speedup Efficiency:\t\t\t%lf%%\n\n", speedup_ratio);
    }

    return 0;
}
