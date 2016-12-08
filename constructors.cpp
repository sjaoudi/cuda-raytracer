#include "structs.cpp"
#include <vector>


using namespace std;
Triangle makeTriangle(float* p1, float* p2, float* p3){
  Triangle tri;
  tri.pt1.x = p1[0];
  tri.pt1.y = p1[1];
  tri.pt1.z = p1[2];

  tri.pt2.x = p2[0];
  tri.pt2.y = p2[1];
  tri.pt2.z = p2[2];

  tri.pt3.x = p3[0];
  tri.pt3.y = p3[1];
  tri.pt3.z = p3[2];
  return tri;


}

__device__ float floatmax(float a, float b){
  if (a > b){return a;}
  return b;
}
