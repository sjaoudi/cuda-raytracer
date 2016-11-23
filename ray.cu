/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include "book.h"
#include "cpu_bitmap.h"
#include "timer.h"
#include <cuda.h>

#define DIM 1024

#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f

struct Point {
  float x, y, z;
  __device__ Point () {}
  __device__ Point (float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

  __device__ Point crossProduct(Point p) {
    Point result;
    result.x = (y*p.z - p.y*z);
    result.y = (p.x*z - x*p.z);
    result.z = (x*p.y - p.x*y);

    return result;
  }
};

__device__ Point operator-(const Point &p1, const Point &p2) {
  Point p3;
  p3.x = p1.x - p2.x;
  p3.y = p1.y - p2.y;
  p3.z = p1.z - p2.z;

  return p3;
}

__device__ Point operator+(const Point &p1, const Point &p2) {
  Point p3;
  p3.x = p1.x + p2.x;
  p3.y = p1.y + p2.y;
  p3.z = p1.z + p2.z;

  return p3;
}

__device__ float operator*(const Point &p1, const Point &p2) {
  float result = p1.x*p2.x + p1.y*p2.y + p1.z*p2.z;
  return result;
}

struct Triangle {
  float r, b, g;
  Point pt1, pt2, pt3;
  __device__ bool leftOf(Point p1, Point p2, Point p3) {

    Point cross = (p2 - p1).crossProduct(p3 - p1);
    Point normal = (pt2 - pt1).crossProduct(pt3 - pt1);
    float dot = cross * normal;

    return dot < 0;

  }
  __device__ bool contains(Point q) {

    return (leftOf(pt2, pt1, q) && leftOf(pt3, pt2, q) && leftOf(pt1, pt3, q));
  }
  __device__ float hit(float ox, float oy, float *n) {

    Point normal = (pt2 - pt1).crossProduct(pt3 - pt1);

    Point o;
    o.x = ox;
    o.y = oy;
    o.z = 0;

    Point d;
    d.x = 0;
    d.y = 0;
    d.z = -1;

    float t = (normal * (pt1 - o)) / (d * normal);

    Point direction = Point(0, 0, -t);
    Point p = o + direction;

    if (!contains(p)) {
      return -INF;
    }

    return t;

  }
};

#define TRIANGLES 1
__constant__ Triangle t[TRIANGLES];

__global__ void kernel(unsigned char *ptr) {
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;
  float ox = (x - DIM / 2);
  float oy = (y - DIM / 2);

  float r = 0, g = 0, b = 0;
  float maxz = -INF;
  for (int i = 0; i < TRIANGLES; i++) {
    float n;
    float htime = t[i].hit(ox, oy, &n);
    if ((ox == 0) && (oy == 0)) {
      printf("htime: %f\n", htime);
      printf("-INF: %f\n", -INF);
      printf("normal: %f\n", (t[i].pt2 - t[i].pt1).crossProduct(t[i].pt3 - t[i].pt1).z);
    }
    if (htime > maxz) {
      float fscale = 1;
      r = t[i].r * fscale;
      g = t[i].g * fscale;
      b = t[i].b * fscale;
      maxz = htime;
    }
  }

  ptr[offset * 4 + 0] = (int)(r * 255);
  ptr[offset * 4 + 1] = (int)(g * 255);
  ptr[offset * 4 + 2] = (int)(b * 255);
  ptr[offset * 4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
  unsigned char *dev_bitmap;
};

int main(void) {
  DataBlock data;
  GPUTimer watch;
  // capture the start time
  watch.start();

  CPUBitmap bitmap(DIM, DIM, &data);
  unsigned char *dev_bitmap;

  // allocate memory on the GPU for the output bitmap
  HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));

  Triangle *temp_tri = (Triangle *)malloc(sizeof(Triangle) * TRIANGLES);

  for (int i = 0; i < TRIANGLES; i++) {
    temp_tri[i].r = rnd(1.0f);
    temp_tri[i].g = rnd(1.0f);
    temp_tri[i].b = rnd(1.0f);

    temp_tri[i].pt1.x = -250;
    temp_tri[i].pt1.y = -250;
    temp_tri[i].pt1.z = 0;

    temp_tri[i].pt2.x = 250;
    temp_tri[i].pt2.y = -250;
    temp_tri[i].pt2.z = 0;

    temp_tri[i].pt3.x = 0;
    temp_tri[i].pt3.y = 250;
    temp_tri[i].pt3.z = 0;
  }

  //HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
  HANDLE_ERROR(cudaMemcpyToSymbol(t, temp_tri, sizeof(Triangle) * TRIANGLES));
  free(temp_tri);

  // generate a bitmap from our sphere data
  dim3 grids(DIM / 16, DIM / 16);
  dim3 threads(16, 16);
  kernel<<<grids, threads>>>(dev_bitmap);

  // copy our bitmap back from the GPU for display
  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(),
                          cudaMemcpyDeviceToHost));

  // get stop time, and display the timing results
  watch.stop();
  printf("Time to generate:  %3.1f ms\n", watch.elapsed());

  HANDLE_ERROR(cudaFree(dev_bitmap));

  // display
  bitmap.display_and_exit(NULL);
}
