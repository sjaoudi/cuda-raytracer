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



  // Initialize the meshes in the scene one by one
  for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
    const aiMesh* paiMesh = scene->mMeshes[i];
    for (unsigned int j = 0; j < paiMesh->mNumVertices; j++) {
      const aiVector3D* pPos = &(paiMesh->mVertices[j]);
      //float v[3];
      //v[0] = pPos->x;
      //v[1] = pPos->y;
      //v[2] = pPos->z;
      std::vector<float> vertex;
      vertex.push_back(pPos->x);
      vertex.push_back(pPos->y);
      vertex.push_back(pPos->z);

      //printf("(%f, %f, %f)\n", v[0], v[1], v[2]);

      vertices.push_back(vertex);
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
