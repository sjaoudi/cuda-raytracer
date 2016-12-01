At this point, we built on the raytracer in the class CUDA examples folder to support triangles. The next step
would be to create complicated models composed of triangles using the class we created. Difficulties so far include compiling
the project (we just copied the entire class examples folder), and debugging in CUDA was also a challenge. The leftOf() function
we created is also confusing us - it appears that it has the appropriate sign flipped (less than instead of greater than), and we 
haven't figured out why. Next steps are implementing light sources (as in the midterm raytracer), and integrating the asset importer.
