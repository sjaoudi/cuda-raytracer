# Final Report

# Project Overview

We set out to create a CUDA raytracer that uses phong lighting and can render object files imported by the assimp asset
importer. We successfully created this project.

# Technical Highlights

First we describe the raytracer. The raytracer constructs a scene object (the same as in the midterm project) and 
an array of lights (as described in the midterm) and triangles (a struct that we define). It passes this data to a CUDA
kernal, which supports full two-dimensional block and thread parallelism.

Within the kernal we reimplemented the midterm raytracer, fully supporting phong lighting. To do this on the graphics card,
we had to write our own structs for triangles, 3D vectors, Rays, Lights, Materials, Views, Scenes, and Hits (all contained
in structs.cpp). We also
implemented crossproduct, dot product, and vector normalization. 

With these building blocks, we reimplemented the hit() function in our triangle struct, including contains() and LeftOf().

The work so far was sufficient to reimplement all of the raytracer from the midterm lab.

Now we describe our asset importer. When our program starts, it calls DoTheImportThing (importer_stuff.cpp) 
on a hardcoded .stl file. Within this function, we accomplish several things.

First, we use assimp's functionality to import a triangulated version of the object. We iterate over meshes and faces
to read all vertices of the triangles. We create an array of triangle structs (which we return). 

Also, we construct the scene struct in this function, so that our objects will be visible. We keep track of the dimensions
of the object that we are trying to render, and set the dimensions of our scene accordingly. By doing this, we can ensure
that all objects will render in the middle of the screen and at a reasonable size (they won't be way off in the corner or 
really tiny). Unfortunately we can make no claims about what direction the figure will face. We hardcode adjustments to the
data that we read in on a per .stl file basis to get the objects to face the positive z direction. 

# References

Cuda By Example
Assimp documentation
http://www.mbsoftworks.sk/index.php?page=tutorials&series=1&tutorial=23 (a helpful guide to using assimp)
Thingiverse.com
We used the class assimp demo code as a starting point for our project, and we obvioulsy reused a lot of code
from our midterm project. 






# Experiemental Results

We successfully rendered three .stl models using our CUDA raytracer. We also piped the triangle data from
these models into input files for our midterm project, which was a traditional CPU-raytracer. Therefore we can
directly compare the preformance of our CUDA raytracer to that of our CPU raytracer. Note that given data from
the corgi and snowman, we did not bother raytracing the kangarphin or kangaroo on the CPU. 

### CPU
| model | time (ms) |  #triangles |
|--------|------| -----|
| corgi.stl |  1573000 (26 min, 13s)   |  3740 |
| snowman.stl |  4260000 (71 min)  | 5558  |
| kangarphin.stl | ?? | 200098 |
| kang.stl |  ?? | 716044 |

Based on the increase in time from the corgi to the snowman, we would expect the kangarphin and kangaroo to take many hours on the CPU.

### CUDA

## Card: GeForce GTX 980 Ti
32x32 blocks, 16x16 threads per block

| model | time (ms) |  #triangles |
|--------|------| -----|
| corgi.stl |   390  | 3740 |
| snowman.stl | 519 | 5558 |
| kangarphin.stl | 27705 (28s) | 200098 |
| kang.stl |   105,000 (1 min, 45s) | 716044 |

So our CUDA raytracer renders corgi.stl 99.98% faster than a traditional CPU raytracer.  

# Instructions
    mkdir build      
    cmake ..    
    make -j8 
    ./ray
To change the .stl file that you render, just open ray.cu and change the hardcoded filename value in the call
to DoTheImportThing. 

