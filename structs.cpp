
#define INF 2e10f
struct vec3 {
  float x, y, z;
  __device__ vec3 () {}
  __device__ vec3 (float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

  __device__ vec3 crossProduct(vec3 p) {
    vec3 result;
    result.x = (y*p.z - p.y*z);
    result.y = (p.x*z - x*p.z);
    result.z = (x*p.y - p.x*y);

    return result;
  }
  __device__ vec3 normalized(){
    float length = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
    return vec3(x/length, y/length, z/length);
  }
};

__device__ vec3 operator-(const vec3 &p1, const vec3 &p2) {
  vec3 p3;
  p3.x = p1.x - p2.x;
  p3.y = p1.y - p2.y;
  p3.z = p1.z - p2.z;

  return p3;
}

__device__ vec3 operator+(const vec3 &p1, const vec3 &p2) {
  vec3 p3;
  p3.x = p1.x + p2.x;
  p3.y = p1.y + p2.y;
  p3.z = p1.z + p2.z;

  return p3;
}

__device__ float operator*(const vec3 &p1, const vec3 &p2) {
  float result = p1.x*p2.x + p1.y*p2.y + p1.z*p2.z;
  return result;
}

__device__ vec3 operator*(const vec3 &p, const float scale) {
  vec3 result = vec3(p.x * scale, p.y * scale, p.z * scale);
  return result;
}
__device__ vec3 operator*(const float scale, const vec3 &p) {
  vec3 result = vec3(p.x * scale, p.y * scale, p.z * scale);
  return result;
}
__device__ vec3 operator/(const vec3 &p, const float scale) {
  vec3 result = vec3(p.x / scale, p.y / scale, p.z / scale);
  return result;
}

struct Ray {
  vec3 origin;
  vec3 direction;
  __device__ vec3 pointAtTime(float t){
    return origin + t*direction;
  }
};

struct Light {
  vec3 position;
  float intensity;
};

struct Material {
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};

struct Triangle {
  float r, b, g;
  vec3 pt1, pt2, pt3;
  Material material;

  __device__ bool leftOf(vec3 p1, vec3 p2, vec3 p3) {

    vec3 cross = (p2 - p1).crossProduct(p3 - p1);
    vec3 normal = (pt2 - pt1).crossProduct(pt3 - pt1);
    float dot = cross * normal;

    return dot < 0;
  }

  __device__ bool contains(vec3 q) {
    return (leftOf(pt2, pt1, q) && leftOf(pt3, pt2, q) && leftOf(pt1, pt3, q));
  }

  __device__ float hit(Ray ray) {

    vec3 normal = (pt2 - pt1).crossProduct(pt3 - pt1);

    vec3 o = ray.origin;
    vec3 d = ray.direction;

    float t = (normal * (pt1 - o)) / (d * normal);

    vec3 p = ray.pointAtTime(t);

    if (!contains(p)) {
      return -INF;
    }

    return t;

  }
  __device__ vec3 normal(){
    return (pt2 - pt1).crossProduct(pt3 - pt1).normalized();
  }
};

struct view_s {
  vec3 eye; // eye position

  vec3 origin; // origin of image plane

  vec3 horiz; // vector from origin bottom left to
  // bottom right side of image plane

  vec3 vert; // vector from origin bottom to
  // top left side of image plane

  vec3 background; // background rgb color in 0-1 range

  int nrows, ncols; // for output image
};

struct scene_s {
  view_s view;
  float ambient;
};

struct Hit {
  float time;
  int shapeID;
};
