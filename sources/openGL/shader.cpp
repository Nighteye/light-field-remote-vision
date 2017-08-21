#include "shader.h"

#include <fstream>
#include <iostream>

#define checkGLErrors() {\
    GLenum error = glGetError(); \
    if(error != GL_NO_ERROR) { \
    std::cout << "GL_ERROR :" << __FILE__ << " "<< __LINE__ << " " << error << std::endl; \
    } \
    }


using namespace glm;

Shader::Shader() : _vertexID(0), _fragmentID(0), _programID(0), _vertexSource(), _fragmentSource()
{
}


Shader::Shader(Shader const &shaderToCopy)
{
    // Copy source files

    _vertexSource = shaderToCopy._vertexSource;
    _fragmentSource = shaderToCopy._fragmentSource;
}


Shader::Shader(std::string vertexSource, std::string fragmentSource) : _vertexID(0), _fragmentID(0), _programID(0),
    _vertexSource(vertexSource), _fragmentSource(fragmentSource)
{
}


Shader::~Shader()
{
    // Destruction du shader

    checkGLErrors();

    if(glIsShader(_vertexID) == GL_TRUE)
        glDeleteShader(_vertexID);

    if(glIsShader(_fragmentID) == GL_TRUE)
        glDeleteShader(_fragmentID);

    if(glIsProgram(_programID) == GL_TRUE)
        glDeleteProgram(_programID);

    checkGLErrors();
}


Shader& Shader::operator=(Shader const &shaderToCopy)
{
    // Copie des fichiers sources

    _vertexSource = shaderToCopy._vertexSource;
    _fragmentSource = shaderToCopy._fragmentSource;

    // Retour du pointeur this

    return *this;
}

bool Shader::add() {

    // Destruction d'un éventuel ancien Shader

    checkGLErrors();

    if(glIsShader(_vertexID) == GL_TRUE)
        glDeleteShader(_vertexID);

    if(glIsShader(_fragmentID) == GL_TRUE)
        glDeleteShader(_fragmentID);

    if(glIsProgram(_programID) == GL_TRUE)
        glDeleteProgram(_programID);

    // Compilation des shaders

    if(!compilerShader(_vertexID, GL_VERTEX_SHADER, _vertexSource)) {

        return false;
    }

    if(!compilerShader(_fragmentID, GL_FRAGMENT_SHADER, _fragmentSource)) {

        return false;
    }

    // Création du programme

    _programID = glCreateProgram();
    assert( glIsProgram(_programID) == GL_TRUE );


    // Association des shaders

    glAttachShader(_programID, _vertexID);
    glAttachShader(_programID, _fragmentID);

    checkGLErrors();

    return true;
}

void Shader::bindAttribLocation( int pos, const char* name ) {

    glBindAttribLocation( _programID, pos, name );
}


bool Shader::link() {

    glLinkProgram(_programID);

    GLint erreurLink;
    glGetProgramiv(_programID, GL_LINK_STATUS, &erreurLink);

    if(erreurLink != GL_TRUE) {

        GLint maxLength;
        glGetShaderiv(_programID,GL_INFO_LOG_LENGTH,&maxLength);
        char* infoLog = (char*)malloc(maxLength);
        glGetShaderInfoLog(_programID,maxLength,NULL, infoLog);
        printf("%s\n", infoLog);
        free(infoLog);
        glDeleteProgram(_programID);
        return false;

    } else {

        return true;
    }
}


bool Shader::compilerShader(GLuint &shader, GLenum type, std::string const &fichierSource)
{
    // Création du shader

    shader = glCreateShader(type);


    // Vérification du shader

    if(shader == 0)
    {
        std::cout << "Erreur, le type de shader (" << type << ") n'existe pas" << std::endl;
        return false;
    }


    // Flux de lecture

    std::ifstream fichier(fichierSource.c_str());


    // Test d'ouverture

    if(!fichier)
    {
        std::cout << "Erreur le fichier " << fichierSource << " est introuvable" << std::endl;
        glDeleteShader(shader);

        return false;
    }


    // Strings permettant de lire le code source

    std::string ligne;
    std::string codeSource;


    // Lecture

    while(getline(fichier, ligne))
        codeSource += ligne + '\n';


    // Fermeture du fichier

    fichier.close();


    // Récupération de la chaine C du code source

    const GLchar* chaineCodeSource = codeSource.c_str();


    // Envoi du code source au shader

    glShaderSource(shader, 1, &chaineCodeSource, 0);


    // Compilation du shader

    glCompileShader(shader);

    GLint isCompiled(0);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);

    if(isCompiled == GL_FALSE) {

        GLint maxLength;
        glGetShaderiv(shader,GL_INFO_LOG_LENGTH,&maxLength);
        char* infoLog = (char*)malloc(maxLength);
        glGetShaderInfoLog(shader,maxLength,NULL, infoLog);
        printf("%s\n", infoLog);
        free(infoLog);
        glDeleteProgram(shader);
        return false;

    } else {

        return true;
    }
}


void Shader::setUniformMat4(std::string name, glm::mat4 matrix) {

    // Localisation de la matrice

    int location = glGetUniformLocation(_programID, name.c_str());

    if(location != -1) {

        glUniformMatrix4fv(location, 1, GL_FALSE, value_ptr(matrix));

    } else {

        printf( "Location FAILED %s\n", name.c_str() );
    }
}

void Shader::setUniformMat4x3( const char* name, glm::mat4x3 matrix ) {

    GLint location = glGetUniformLocation( _programID,(const GLchar*)name );
    if(location != -1) {
        glUniformMatrix4x3fv( location, 1, GL_FALSE, value_ptr(matrix) );
    } else {
        printf( "Location FAILED %s\n", name );
    }
}

void Shader::setUniformMat3( const char* name, glm::mat3 matrix ) {

    GLint location = glGetUniformLocation( _programID,(const GLchar*)name );
    if(location != -1) {
        glUniformMatrix3fv( location, 1, GL_FALSE, value_ptr(matrix) );
    } else {
        printf( "Location FAILED %s\n", name );
    }
}

void Shader::setUniform3fv( const char* name, glm::vec3 vector ) {

    GLint location = glGetUniformLocation( _programID,(const GLchar*)name );
    if(location != -1) {
        glUniform3fv( location, 1, value_ptr(vector) );
    } else {
        printf( "Location FAILED %s\n", name );
    }
}

void Shader::setUniform3fv(const char* name, float value[3]) {

    GLint location = glGetUniformLocation( _programID,(const GLchar*)name );
    if(location != -1) {
        glUniform3fv( location, 1, value );
    } else {
        printf( "Location FAILED %s\n", name );
    }
}

void Shader::setUniformi(const char* name, GLint value) {

    GLint location = glGetUniformLocation( _programID,(const GLchar*)name );
    if(location != -1) {
        glUniform1i( location, value );
    } else {
        printf( "Location FAILED %s\n", name );
    }
}

void Shader::setUniformf(const char* name, GLfloat value) {

    GLint location = glGetUniformLocation( _programID,(const GLchar*)name );
    if(location != -1) {
        glUniform1f( location, value );
    } else {
        printf( "Location FAILED %s\n", name );
    }
}

void Shader::setUniformArrayf( const char* name, int size, const GLfloat *values ) {

    GLint location = glGetUniformLocation( _programID, (const GLchar*)name );

    if( location != -1 ) {

        glUniform1fv( location, size, values );

    } else {

        printf( "Location FAILED %s\n", name );
    }
}


// Getter

GLuint Shader::getProgramID() const {

    return _programID;
}

// -------------------------------------------- CLASS WITH GEOMETRY SHADER -------------------------------------------- //

ShaderGeometry::ShaderGeometry() : Shader(), _geometryID(0) {}

ShaderGeometry::ShaderGeometry( ShaderGeometry const &shaderToCopy ) : Shader(shaderToCopy) {

    _geometrySource = shaderToCopy._geometrySource;
}

ShaderGeometry::ShaderGeometry( std::string vertexSource, std::string fragmentSource, std::string geometrySource ) :
    Shader( vertexSource, fragmentSource ), _geometryID(0), _geometrySource(geometrySource) {}


ShaderGeometry::~ShaderGeometry() {

    glDeleteShader(_vertexID);
    glDeleteShader(_fragmentID);
    glDeleteShader(_geometryID);
    glDeleteProgram(_programID);
}

ShaderGeometry& ShaderGeometry::operator=(ShaderGeometry const &shaderToCopy) {

    _vertexSource = shaderToCopy._vertexSource;
    _fragmentSource = shaderToCopy._fragmentSource;
    _geometrySource = shaderToCopy._geometrySource;

    return *this;
}

// also add the geometry shader
bool ShaderGeometry::add() {

    checkGLErrors();

    if(glIsShader(_vertexID) == GL_TRUE)
        glDeleteShader(_vertexID);

    if(glIsShader(_fragmentID) == GL_TRUE)
        glDeleteShader(_fragmentID);

    if(glIsShader(_geometryID) == GL_TRUE)
        glDeleteShader(_geometryID);

    if(glIsProgram(_programID) == GL_TRUE)
        glDeleteProgram(_programID);

    if(!compilerShader(_vertexID, GL_VERTEX_SHADER, _vertexSource)) {

        return false;
    }

    if(!compilerShader(_fragmentID, GL_FRAGMENT_SHADER, _fragmentSource)) {

        return false;
    }

    if(!compilerShader(_geometryID, GL_GEOMETRY_SHADER, _geometrySource)) {

        return false;
    }

    _programID = glCreateProgram();
    assert( glIsProgram(_programID) == GL_TRUE );

    glAttachShader(_programID, _vertexID);
    glAttachShader(_programID, _fragmentID);
    glAttachShader(_programID, _geometryID);

    checkGLErrors();

    return true;
}
