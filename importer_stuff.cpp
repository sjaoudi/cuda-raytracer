#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

#include "constructors.cpp"

vector<Triangle> DoTheImportThing( const std::string& pFile)
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
  // printf("Yay! it worked\n");





  for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
    aiMesh* mesh = scene->mMeshes[i];
    for (unsigned int j = 0; j < mesh->mNumFaces; j++){
      const aiFace& face = mesh->mFaces[j];
      for (int k = 0; k < 3; k++){
        aiVector3D pos = mesh->mVertices[face.mIndices[k]];
        std::vector<float> vertex;
        vertex.push_back(pos.x);
        vertex.push_back(pos.z - 3);
        vertex.push_back(-pos.y);
        vertices.push_back(vertex);
      }


    }
  }

  //for (unsigned int i = 0; i < vertices.size(); i++) {
  //  printf("(%f, %f, %f)\n", vertices[i][0], vertices[i][1], vertices[i][2]);
  //}

  for (unsigned int i = 0; i < vertices.size()-2; i += 3){
    Triangle tri = makeTriangle(vertices[i], vertices[i+1], vertices[i+2]);
    tris.push_back(tri);
  }
  return tris;

  // We're done. Everything will be cleaned up by the importer destructor

}
