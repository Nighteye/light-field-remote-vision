#ifndef SHADER_H
#define SHADER_H

#include <GL/glew.h>

// Includes GLM
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>

class Shader {

public:

    Shader();
    Shader(Shader const &shaderToCopy);
    Shader(std::string vertexSource, std::string fragmentSource);
    ~Shader();

    Shader& operator=(Shader const &shaderToCopy);

    bool add();
    void bindAttribLocation( int pos, const char* name );
    bool link();
    bool compilerShader(GLuint &shader, GLenum type, std::string const &fichierSource);

    void setUniformMat4( std::string name, glm::mat4 matrix );
    void setUniformMat4x3( const char* name, glm::mat4x3 matrix );
    void setUniformMat3( const char* name, glm::mat3 matrix );
    void setUniform3fv( const char* name, glm::vec3 vector );
    void setUniform3fv( const char* name, float value[3] );
    void setUniformi( const char* name, GLint value );
    void setUniformArrayf( const char* name, int size, const GLfloat *values );
    void setUniformf( const char* name, GLfloat value );
    GLuint getProgramID() const;

protected:

    GLuint _vertexID;
    GLuint _fragmentID;
    GLuint _programID;

    std::string _vertexSource;
    std::string _fragmentSource;
};

// ShaderGeometry is Shader with geometry shader appended to it

class ShaderGeometry : public Shader {

public:

    ShaderGeometry();
    ShaderGeometry(ShaderGeometry const &shaderToCopy);
    ShaderGeometry(std::string vertexSource, std::string fragmentSource, std::string geometrySource);
    ~ShaderGeometry();

    ShaderGeometry& operator=(ShaderGeometry const &shaderToCopy);
    bool add();

private:

    GLuint _geometryID;
    std::string _geometrySource;

};

#endif /* #ifndef SHADER_H */
