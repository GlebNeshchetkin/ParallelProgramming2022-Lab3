#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <utility>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


//#define dd double

__device__
const double G = 6.674e-11;
__device__
const double dt = 0.001;

struct MatPoint {
    double x;
    double y;
    double vx;
    double vy;
    double m;
};

__global__
void calcForce(double* X, MatPoint* device_points, int num_of_points, int partition_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int start = i * partition_size;
    int end = (i + 1) * partition_size;
    //printf("%d-%d\n", start, end);
    for (int i = start; i < end; i++) {
        //for (unsigned i = 0; i < num_of_points; i += 1) {
            double sum_x = 0;
            double sum_y = 0;
            double x_i = static_cast<MatPoint>(device_points[i]).x;
            double y_i = static_cast<MatPoint>(device_points[i]).y;
            double m_i = static_cast<MatPoint>(device_points[i]).m;
            //printf("%f-%f-%f\n", x_i, y_i, m_i);
            for (unsigned j = 0; j < num_of_points; ++j) {
                double x_j = static_cast<MatPoint>(device_points[j]).x;
                double y_j = static_cast<MatPoint>(device_points[j]).y;
                double m_j = static_cast<MatPoint>(device_points[j]).m;
                if (i == j) {
                    continue;
                }

                double dist = sqrt(pow((x_j - x_i), 2) + pow((y_j - y_i), 2));
                sum_x += m_j * (x_j - x_i) / pow(dist, 3);
                sum_y += m_j * (y_j - y_i) / pow(dist, 3);
            }

            X[i * 2] = (G * m_i * sum_x);
            X[i * 2 + 1] = (G * m_i * sum_y);
            //printf("%f-%f\n", X[i*2], X[i*2+1]);
        //}
    }
}

__global__
void simulationStep_device(MatPoint* device_points, double* X, int num_of_points) {
    //printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    for (unsigned i = 0; i < num_of_points; i += 1) {
        double x_i = static_cast<MatPoint>(device_points[i]).x;
        double y_i = static_cast<MatPoint>(device_points[i]).y;
        double vx_i = static_cast<MatPoint>(device_points[i]).vx;
        double vy_i = static_cast<MatPoint>(device_points[i]).vy;
        double m_i = static_cast<MatPoint>(device_points[i]).m;
        //printf("->->%f<-<-\n", static_cast<MatPoint>(device_points[i]).x);
        device_points[i].vx += X[i * 2] / m_i * dt;
        device_points[i].vy += X[i * 2 + 1] / m_i * dt;
        device_points[i].x += vx_i * dt;
        device_points[i].y += vy_i * dt;
        //printf("->->->->%f<-<-<-<-\n", static_cast<MatPoint>(device_points[i]).x-x_i);
        //printf("->->->->%f<-<-<-<-\n", vx_i);
    }
}

void read_file_thrust(thrust::device_vector<MatPoint>& points) {
    std::ifstream file("input.txt");
    double x, y, vx, vy, m;
    while (!file.eof()) {
        file >> x >> y >> vx >> vy >> m;
        points.push_back({ x, y, vx, vy, m });
        //printf("%f<->%f<->%f<->%f<->%f\n", x, y, vx, vy, m);
    }
}

void print_results(std::ofstream& file, MatPoint* points, int num_of_points) {
    // std::ofstream file("output.txt");
    for (int i = 0; i < num_of_points; i++) {
        file << points[i].x << "," << points[i].y << ", ";
    }
    file << "\n";
}

int main() {
    std::vector<MatPoint> points;
    thrust::device_vector<MatPoint> device_points;
    thrust::device_vector<MatPoint> device_points0;
    thrust::device_vector<double> results(1000);

    read_file_thrust(device_points);
    MatPoint* device_points_pointers = thrust::raw_pointer_cast(device_points.data());
    MatPoint* device_points_pointers0 = (MatPoint*)malloc(device_points.size() * sizeof(MatPoint));

    double* device_results_pointers = thrust::raw_pointer_cast(results.data());
    double* results_pointers;
    results_pointers = (double*)malloc(1000 * sizeof(double));
    int num_of_points = device_points.size();

    std::ofstream file("output.txt");
    file << "t,";
    for (unsigned i = 0; i < device_points.size(); ++i) {
        file << "x" << i + 1 << ",y" << i + 1 << ",";
    }
    file << "\n";

    int num_of_threads = 32;
    int partition_size = num_of_points / num_of_threads;
    double time_sum = 0;
    double t = 0;
    int st = 0;
    while (t < 100) {
        st++;
        printf("t=%f\n", t);
        clock_t start = clock();
        calcForce<<<1, num_of_threads>>>(device_results_pointers, device_points_pointers, num_of_points, partition_size);
        simulationStep_device<<<1, 1>>>(device_points_pointers, device_results_pointers, num_of_points);
        //cudaMemcpy(results_pointers, device_results_pointers, 1000 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(device_points_pointers0, device_points_pointers, num_of_points * sizeof(MatPoint), cudaMemcpyDeviceToHost);
        clock_t end = clock();
        time_sum += ((double)(end - start)) / CLOCKS_PER_SEC;
        print_results(file, device_points_pointers0, num_of_points);
        t += dt;
    }
    printf("Num of threads: %d, Time taken: %.10f\n", num_of_threads, time_sum/st);
    
    return 0;
}
