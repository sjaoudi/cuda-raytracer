# Final Project Proposal

 Overview

We're going to update our raytracer from the midterm project to run with the
CUDA framework. We think we can drastically improve speed by parallelizing the
pixel grid, and we're curious how fast the speedup can get. We both enjoyed
using CUDA in the class, and we want to explore more uses of the framework.
We'll also apply our CUDA program to a scene with pre-made objects in .obj or
.stl format. These objects will be parsed by a program like assimp.

# Course Concepts Used

Raytracing
Phong lighting model
CUDA in 2D

# Other software Tools

.obj and .stl file parser - assimp
Website such as thingverse.com for obtaining files

# Goals

Short-term:
Program that parses and renders existing scenes from midterm project, with CUDA.
This requires re-implementation of existing vector methods in CUDA language
framework.
With object files, we want the ability to find the normal, given a point on its
surface.

Long-term:
Working, highly-parallelized raytracer in CUDA.
Ability to raytrace a scene without writing classes for each shape.

Reach:
Other materials, such as transparency

# References

CUDA Documentation
CUDA By Example book
Object files from thingverse.com

# Assessment

What would make your project a success?

At the least, a working CUDA raytacer. Hopefully we are also able to create our
own scenes with custom objects.
