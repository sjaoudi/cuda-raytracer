#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

#include "constructors.cpp"

vector<Triangle> DoTheImportThing( const std::string& pFile, scene_s *temp_scene)
{
  // Create an instance of the Importer class
  Assimp::Importer importer;
  // And have it read the given file with some example postprocessing
  // Usually - if speed is not the most important aspect for you - you'll
  // propably to request more postprocessing than we do in this example.
  const aiScene* scene = importer.ReadFile( pFile,
        aiProcess_Triangulate            |
        aiProcess_JoinIdenticalVertices  |
        aiProcess_SortByPType
      );

  std::vector<std::vector<float>> vertices;
  vector<Triangle> tris;
  // If the import failed, report it
  if( !scene)
  {
    printf("Failed to import scene\n");
    return tris;
  }
  // Now we can access the file's contents.
  printf("Yay! it worked\n");

  for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
    aiMesh* mesh = scene->mMeshes[i];
    for (unsigned int j = 0; j < mesh->mNumFaces; j++){
      const aiFace& face = mesh->mFaces[j];
      for (int k = 0; k < 3; k++){
        aiVector3D pos = mesh->mVertices[face.mIndices[k]];
        std::vector<float> vertex;
        // vertex.push_back(pos.y);
        // vertex.push_back(pos.z);
        // vertex.push_back(pos.x);

        vertex.push_back(pos.x);
        vertex.push_back(pos.z);
        vertex.push_back(-pos.y);


        vertices.push_back(vertex);
      }
    }
  }

  float x = 0;
  float y = 0;
  float z = 0;

  float left = INF;
  float right = -INF;
  float low = INF;
  float high = -INF;
  float near = -INF;
  float far = INF;

  for (unsigned int i = 0; i < vertices.size(); i++) {
    //printf("(%f, %f, %f)\n", vertices[i][0], vertices[i][1], vertices[i][2]);
    x += vertices[i][0];
    y += vertices[i][1];
    z += vertices[i][2];

    if (vertices[i][0] < left) {
      left = vertices[i][0];
    }
    if (vertices[i][0] > right) {
      right = vertices[i][0];
    }
    if (vertices[i][1] < low) {
      low = vertices[i][1];
    }
    if (vertices[i][1] > high) {
      high = vertices[i][1];
    }
    if (vertices[i][2] < far) {
      far = vertices[i][2];
    }
    if (vertices[i][2] > near) {
      near = vertices[i][2];
    }
  }
  float horiz = 2*fabs(right - left);
  float vert = 2*fabs(high - low);
  float larger = std::fmax(horiz, vert);

  float dist = fabs(near - far);

  temp_scene->view.horiz.x = larger;
  temp_scene->view.horiz.y = 0;
  temp_scene->view.horiz.z = 0;

  temp_scene->view.vert.x = 0;
  temp_scene->view.vert.y = larger;
  temp_scene->view.vert.z = 0;

  temp_scene->view.eye.x = x / vertices.size();
  temp_scene->view.eye.y = y / vertices.size() - 0;
  temp_scene->view.eye.z = z / vertices.size() + 4*dist;

  temp_scene->view.origin.x = x / vertices.size() - temp_scene->view.horiz.x / 2;
  temp_scene->view.origin.y = y / vertices.size() - temp_scene->view.vert.y / 2;
  temp_scene->view.origin.z =  z / vertices.size() + dist;


  for (unsigned int i = 0; i < vertices.size()-2; i += 3){
    Triangle tri = makeTriangle(vertices[i], vertices[i+1], vertices[i+2]);
    //printf("triangle %f %f %f %f %f %f %f %f %f\n", vertices[i][0], vertices[i][1], vertices[i][2],
    //                                                vertices[i+1][0], vertices[i+1][1], vertices[i+1][2],
    //                                                vertices[i+2][0], vertices[i+2][1], vertices[i+2][2]);
    tris.push_back(tri);
  }

  //printf("number of triangles: %d\n", tris.size());
  return tris;

  // We're done. Everything will be cleaned up by the importer destructor

}
