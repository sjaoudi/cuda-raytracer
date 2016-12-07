#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>


bool DoTheImportThing( const std::string& pFile)
{
  // Create an instance of the Importer class
  Assimp::Importer importer;
  // And have it read the given file with some example postprocessing
  // Usually - if speed is not the most important aspect for you - you'll
  // propably to request more postprocessing than we do in this example.
  const aiScene* scene = importer.ReadFile( pFile,
        aiProcess_Triangulate            |
        aiProcess_JoinIdenticalVertices  |
        aiProcess_SortByPType);

  // If the import failed, report it
  if( !scene)
  {
    printf("oops. it didn't work\n");
    return false;
  }
  // Now we can access the file's contents.
  printf("Yay! it worked\n");

  std::vector<float*> vertices;

  // Initialize the meshes in the scene one by one
  for (unsigned int i = 0 ; i < scene->mNumMeshes; i++) {
    const aiMesh* paiMesh = scene->mMeshes[i];
    for (unsigned int i = 0 ; i < paiMesh->mNumVertices ; i++) {
      const aiVector3D* pPos = &(paiMesh->mVertices[i]);
      float v[3];
      v[0] = pPos->x;
      v[1] = pPos->y;
      v[2] = pPos->z;

      printf("(%f, %f, %f)\n", v[0], v[1], v[2]);

      vertices.push_back(v);
    }
  }

  // We're done. Everything will be cleaned up by the importer destructor
  return true;
}


int main(){
  std::string fname("corgi.stl");
  return DoTheImportThing(fname);
}
