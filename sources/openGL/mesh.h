#ifndef MESH_H
#define MESH_H

#include <vector>
#include <string>
#include <GL/glew.h>
#include <iostream>

// Includes GLM
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "shader.h"

// Macro for VBO
#ifndef BUFFER_OFFSET

#define BUFFER_OFFSET(offset) ((char*)NULL + (offset))

#endif

#define checkGLErrors() {\
    GLenum error = glGetError(); \
    if(error != GL_NO_ERROR) { \
    std::cout << "GL_ERROR :" << __FILE__ << " "<< __LINE__ << " " << error << std::endl; \
    } \
    }

class InputView;

class PointCloud {

public:

    PointCloud( const std::vector<GLfloat> &vertices, const std::vector<GLfloat> &normals );
    ~PointCloud();

    std::vector<GLfloat> getVertices() const;
    std::vector<GLfloat> getNormals() const;
    GLuint getVAO();

protected:

    GLuint _vboID, _vaoID;
    std::vector<GLfloat> _vertices;
    std::vector<GLfloat> _normals;
    uint _verticesSize;
    uint _normalsSize;

private:

    void load();
};

class Mesh : public PointCloud {

public:

    Mesh( const std::vector<GLfloat> &vertices, const std::vector<GLfloat> &normals, const std::vector<GLuint> &triangles );
    ~Mesh();

    void display( const glm::mat4 &renderMatrix, InputView *currentCamera = 0 );
    bool isMeshOK();
    void initTriangleMeshShader( Shader *shader );
    void initTextureMappingShader( Shader *shader );
    uint getNbTriangles();

private:

    void load();

    Shader _triangleMeshShader;
    Shader _textureMappingShader;
    GLuint _eboID;
    std::vector<GLuint> _triangles;
    uint _trianglesSize;
};

#endif /* #ifndef MESH_H */
