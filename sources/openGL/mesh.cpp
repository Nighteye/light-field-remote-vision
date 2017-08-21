#include "mesh.h"
#include "view.h"

PointCloud::PointCloud( const std::vector<GLfloat> &vertices, const std::vector<GLfloat> &normals ) :

    _vboID(0),
    _vaoID(0),
    _vertices(vertices),
    _normals(normals) {

    _verticesSize =  _vertices.size() * sizeof(GLfloat);
    _normalsSize =  _normals.size() * sizeof(GLfloat);

    checkGLErrors();

    PointCloud::load();

    checkGLErrors();
}

PointCloud::~PointCloud( ) {

    glDeleteBuffers(1, &_vboID);
    glDeleteVertexArrays(1, &_vaoID);
}

void PointCloud::load( ) {

    checkGLErrors();

    if(glIsVertexArray(_vaoID) == GL_TRUE) {
        glDeleteVertexArrays(1, &_vaoID);
    }
    if(glIsBuffer(_vboID) == GL_TRUE) {
        glDeleteBuffers(1, &_vboID);
    }
    glGenVertexArrays(1, &_vaoID);
    glGenBuffers(1, &_vboID);

    glBindVertexArray(_vaoID);

    glBindBuffer(GL_ARRAY_BUFFER, _vboID);
    glBufferData(GL_ARRAY_BUFFER, _verticesSize + _normalsSize, 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, _verticesSize, _vertices.data());
    glBufferSubData(GL_ARRAY_BUFFER, _verticesSize, _normalsSize, _normals.data());

    // Vertex Positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
    glEnableVertexAttribArray(0);

    // Normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(_verticesSize));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    checkGLErrors();
}

std::vector<GLfloat> PointCloud::getVertices() const {

    return _vertices;
}

std::vector<GLfloat> PointCloud::getNormals() const {

    return _normals;
}

GLuint PointCloud::getVAO() {

    return _vaoID;
}

Mesh::Mesh( const std::vector<GLfloat> &vertices, const std::vector<GLfloat> &normals, const std::vector<GLuint> &triangles ) :

    PointCloud( vertices, normals ),
    _triangleMeshShader( "sources/openGL/shaders/triangleMesh.vert", "sources/openGL/shaders/triangleMesh.frag" ),
    _textureMappingShader( "sources/openGL/shaders/textureMapping.vert", "sources/openGL/shaders/textureMapping.frag" ),
    _eboID(0),
    _triangles(triangles) {

    _trianglesSize = _triangles.size() * sizeof(GLuint);

    checkGLErrors();

    initTriangleMeshShader( &_triangleMeshShader );
    initTextureMappingShader( &_textureMappingShader );

    checkGLErrors();

    load();
}

Mesh::~Mesh( ) {

    glDeleteBuffers(1, &_vboID);
    glDeleteBuffers(1, &_eboID);
    glDeleteVertexArrays(1, &_vaoID);
}

void Mesh::load( ) {

    checkGLErrors();

    if(glIsVertexArray(_vaoID) == GL_TRUE) {
        glDeleteVertexArrays(1, &_vaoID);
    }
    if(glIsBuffer(_vboID) == GL_TRUE) {
        glDeleteBuffers(1, &_vboID);
    }
    if(glIsBuffer(_eboID) == GL_TRUE) {
        glDeleteBuffers(1, &_eboID);
    }
    glGenVertexArrays(1, &_vaoID);
    glGenBuffers(1, &_vboID);
    glGenBuffers(1, &_eboID);

    glBindVertexArray(_vaoID);

    glBindBuffer(GL_ARRAY_BUFFER, _vboID);
    glBufferData(GL_ARRAY_BUFFER, _verticesSize + _normalsSize, 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, _verticesSize, _vertices.data());
    glBufferSubData(GL_ARRAY_BUFFER, _verticesSize, _normalsSize, _normals.data());

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _eboID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, _trianglesSize, _triangles.data(), GL_STATIC_DRAW);

    // Vertex Positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
    glEnableVertexAttribArray(0);

    // Normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(_verticesSize));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    checkGLErrors();
}

bool Mesh::isMeshOK() {

    return glIsProgram(_triangleMeshShader.getProgramID()) == GL_TRUE;
    return glIsProgram(_textureMappingShader.getProgramID()) == GL_TRUE;
}

void Mesh::display( const glm::mat4 &renderMatrix, InputView *currentCamera ) {

    checkGLErrors();

    // ----------------------- TEXTURE MAPPING ----------------------- //

    if( currentCamera != 0 ) {

        glEnable( GL_DEPTH_TEST );
        glCullFace( GL_BACK );
        glEnable( GL_CULL_FACE );

        glUseProgram( _textureMappingShader.getProgramID() );

        glBindVertexArray( _vaoID );

        _triangleMeshShader.setUniformMat4( "modelviewProjection", renderMatrix );
        // send current camera parameters and image to texture mapping shader
        currentCamera->setTexMappingUniform( &_textureMappingShader );

        glBindTexture(GL_TEXTURE_RECTANGLE, currentCamera->getTextureID());

        glDrawElements(GL_TRIANGLES, _triangles.size(), GL_UNSIGNED_INT, 0 );

        glBindTexture(GL_TEXTURE_RECTANGLE, 0);

        glBindVertexArray(0);

        glUseProgram(0);

        glDisable( GL_CULL_FACE );
        glDisable( GL_DEPTH_TEST );
    }

    // --------------------- WIRED TRIANGULATION --------------------- //

    if(false) { // TODO boolean

        glEnable( GL_LINE_SMOOTH );
        glEnable( GL_POLYGON_SMOOTH );
        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
        glEnable( GL_DEPTH_TEST );
        glCullFace( GL_BACK );
        glEnable( GL_CULL_FACE );

        glUseProgram(_triangleMeshShader.getProgramID());

        glBindVertexArray(_vaoID);

        GLfloat in_color[] = {0.,0.,1.};
        _triangleMeshShader.setUniform3fv("in_color", in_color);
        _triangleMeshShader.setUniformMat4( "modelviewProjection", renderMatrix );

        // count is the number of vertex making the triangles : N triangles = 3* vertex
        glDrawElements(GL_TRIANGLES, _triangles.size(), GL_UNSIGNED_INT, 0);

        glBindVertexArray(0);

        glUseProgram(0);

        glDisable( GL_CULL_FACE );
        glDisable( GL_DEPTH_TEST );
        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
        glDisable( GL_POLYGON_SMOOTH );
        glDisable( GL_LINE_SMOOTH );
    }

    checkGLErrors();
}

void Mesh::initTriangleMeshShader( Shader *shader ) {

    checkGLErrors();

    assert( shader->add() );

    // Bind shader inputs
    shader->bindAttribLocation( 0, "in_position" );

    assert( shader->link() );

    checkGLErrors();
}

void Mesh::initTextureMappingShader( Shader *shader ) {

    checkGLErrors();

    assert( shader->add() );

    // Bind shader inputs
    shader->bindAttribLocation( 0, "in_position" );
    shader->bindAttribLocation( 2, "myTexture" );

    assert( shader->link() );

    checkGLErrors();
}

uint Mesh::getNbTriangles() {

    return _trianglesSize;
}

