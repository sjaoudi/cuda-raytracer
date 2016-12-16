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
#include <cmath>
#include "importer_stuff.cpp"

#define DIM 1024
#define rnd(x) (x * rand() / RAND_MAX)
#define LIGHTS 4

__constant__ Light l[LIGHTS];
__constant__ scene_s scene;

__device__ vec3 convertColor(vec3 color) {
  float cmax = color.x;
  vec3 scaled;

  if (color.y > cmax) { cmax = color.y; }
  if (color.z > cmax) { cmax = color.z; }
  if (cmax > 1) {
    scaled = color / cmax;
  }
  else {
    scaled = color;
  }

  vec3 result;
  result.x = (int)(233 * scaled.x);
  result.y = (int)(233 * scaled.y);
  result.z = (int)(233 * scaled.z);

  return result;
}

__device__ vec3 convertToWorld(int col, int row) {
  float pixelwidth = float(scene.view.horiz.x)/scene.view.ncols;
  float pixelheight = float(scene.view.vert.y)/scene.view.nrows;

  float worldx = scene.view.origin.x + col*pixelwidth;
  float worldy = scene.view.origin.y + row*pixelheight;
  float worldz = scene.view.origin.z;

  return vec3(worldx, worldy, worldz);
}

__device__ Ray makeRay(int col, int row) {
  vec3 p = convertToWorld(col, row);
  Ray ray;
  ray.origin = scene.view.eye;
  ray.direction = p - scene.view.eye;

  return ray;
}

__device__ Hit findClosest(Ray ray, Triangle* triangles, int count) {
  Hit closest;
  closest.shapeID = -1;
  closest.time = 0;

  for (int i = 0; i < count; i++) {
    float t = triangles[i].hit(ray);
    if (t > 0.001) {
      if (closest.shapeID == -1 || t < closest.time) {
        closest.shapeID = i;
        closest.time = t;
      }
    }
  }

  return closest;
}

__device__ bool isInShadow(Light L, Triangle tri, vec3 p, Triangle* triangles, int count){
  Ray ray;
  ray.origin = L.position;
  ray.direction = p - L.position;
  Hit closest = findClosest(ray, triangles, count);
  float t = closest.time;
  if (t != tri.hit(ray) || t == -1){
    return true;
  }
  return false;

}

__device__ vec3 phong(Ray ray, Light L, Triangle triangle){

  vec3 diffuse, specular;
  float t = triangle.hit(ray);
  vec3 p = ray.pointAtTime(t);
  int alpha = 4;

  vec3 l = (L.position - p).normalized();
  vec3 n = triangle.normal();
  diffuse = triangle.material.diffuse * floatmax(l*n, 0.0);

  vec3 r = (2*(l*n)*n - l).normalized();
  vec3 v = (-1*ray.direction).normalized();
  specular = triangle.material.specular * pow(floatmax(r*v, 0.0), alpha);

  return L.intensity * (diffuse + specular);

}

__device__ vec3 doLighting(Ray ray, Triangle triangle, Triangle* triangles, int count) {

  vec3 color = scene.ambient * triangle.material.ambient;

  for (int i = 0; i < LIGHTS; i++) {
    Light L = l[i];
    vec3 p = ray.pointAtTime(triangle.hit(ray));
    if (!isInShadow(L, triangle, p, triangles, count)) {
      color = color + phong(ray, L, triangle);
    }
  }
  return convertColor(color);
}

__device__ vec3 traceRay(Ray ray, Triangle* triangles, int count) {
  Hit closest = findClosest(ray, triangles, count);
  vec3 background = scene.view.background;

  vec3 color = convertColor(background);
  if (closest.shapeID != -1) {
    color = doLighting(ray, triangles[closest.shapeID], triangles, count);
  }

  return color;
}

__global__ void kernel(unsigned char *ptr, Triangle* triangles, int count) {

  int y = threadIdx.y + blockIdx.y * blockDim.y;
  while (y < scene.view.nrows) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    while (x < scene.view.ncols) {
      Ray ray = makeRay(x, y);
      vec3 color =  traceRay(ray, triangles, count);

      // map from threadIdx/BlockIdx to pixel position
      int offset = x + y * scene.view.ncols;

      ptr[offset * 4 + 0] = (int)(color.x);
      ptr[offset * 4 + 1] = (int)(color.y);
      ptr[offset * 4 + 2] = (int)(color.z);
      ptr[offset * 4 + 3] = 233;

      x += gridDim.x * blockDim.x;
    }
    y += gridDim.y * blockDim.y;
  }
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

  scene_s *temp_scene = new scene_s;

  Triangle *triangles;
  std::vector<Triangle> model_triangles = DoTheImportThing("corgi.stl", temp_scene);
  int count = model_triangles.size();

  // allocate memory on the GPU for the output bitmap
  HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));
  HANDLE_ERROR(cudaMalloc((void **)&triangles, sizeof(Triangle) * count));

  Triangle *temp_tri = (Triangle *)malloc(sizeof(Triangle) * count);
  Light *temp_lights = (Light * )malloc(sizeof(Light) * LIGHTS);

  // temp_scene->view.eye.x = 0;
  // temp_scene->view.eye.y = 0;
  // temp_scene->view.eye.z = 20;
  //
  // temp_scene->view.origin.x = -5;
  // temp_scene->view.origin.y = -2;
  // temp_scene->view.origin.z = 8;
  //
  // temp_scene->view.horiz.x = 10;
  // temp_scene->view.horiz.y = 0;
  // temp_scene->view.horiz.z = 0;
  //
  // temp_scene->view.vert.x = 0;
  // temp_scene->view.vert.y = 10;
  // temp_scene->view.vert.z = 0;

  temp_scene->view.background.x = 0;
  temp_scene->view.background.y = 0;
  temp_scene->view.background.z = 0;

  temp_scene->view.nrows = DIM;
  temp_scene->view.ncols = DIM;

  temp_scene->ambient = 0.1;

  for (int i = 0; i < count; i++) {

    temp_tri[i] = model_triangles[i];

    Material material;
    material.ambient.x = 0;
    material.ambient.y = 0;
    material.ambient.z = 1;

    material.diffuse.x = 0;
    material.diffuse.y = 0;
    material.diffuse.z = 1;

    material.specular.x = 1;
    material.specular.y = 1;
    material.specular.z = 1;

    temp_tri[i].material = material;
  }

  temp_lights[0].position = temp_scene->view.eye;
  temp_lights[0].intensity = 1;

  temp_lights[1].position.x = 5;
  temp_lights[1].position.y = 0;
  temp_lights[1].position.z = 5;
  temp_lights[1].intensity = 1;

  temp_lights[2].position.x = -5;
  temp_lights[2].position.y = 0;
  temp_lights[2].position.z = 5;
  temp_lights[2].intensity = 1;

  temp_lights[3].position.x = -10;
  temp_lights[3].position.y = -10;
  temp_lights[3].position.z = -10;
  temp_lights[3].intensity = 1;

  HANDLE_ERROR(cudaMemcpyToSymbol(scene, temp_scene, sizeof(scene_s)));
  HANDLE_ERROR(cudaMemcpyToSymbol(l, temp_lights, sizeof(Light) * LIGHTS));
  HANDLE_ERROR(cudaMemcpy(triangles, temp_tri, sizeof(Triangle) * count, cudaMemcpyHostToDevice));

  free(temp_lights);
  free(temp_tri);
  free(temp_scene);

  // generate a bitmap from our sphere data
  dim3 grids(32, 32);
  dim3 threads(25, 25);
  kernel<<<grids, threads>>>(dev_bitmap, triangles, count);
  // copy our bitmap back from the GPU for display
  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

  // get stop time, and display the timing results
  watch.stop();
  printf("Time to generate:  %3.1f ms\n", watch.elapsed());

  HANDLE_ERROR(cudaFree(dev_bitmap));
  HANDLE_ERROR(cudaFree(triangles));

  // display
  bitmap.display_and_exit(NULL);
}
