#include <GL/glew.h>

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "shaderHandler.h"

#define checkGLErrors() {\
    GLenum error = glGetError(); \
    if(error != GL_NO_ERROR) { \
        std::cout << "GL_ERROR :" << __FILE__ << " "<< __LINE__ << " " << error << std::endl; \
    } \
}

ShaderHandler::ShaderHandler() : _shader(0),_vertexAttached(false),_fragmentAttached(false) {}

ShaderHandler::~ShaderHandler() {

    if(_vertexAttached){
        glDetachShader(_shader,_vertex);
        glDeleteShader(_vertex);
    }
    if(_fragmentAttached){
        glDetachShader(_shader,_fragment);
        glDeleteShader(_fragment);
    }
    if(_shader!=0) {
        glDeleteProgram(_shader);
    }
}

bool ShaderHandler::addShader(ShaderType type,const char* src) {

    if(_shader == 0){
        _shader = glCreateProgram();
    }

    GLuint shader = 0;
    if(type == Vertex){
        _vertex = glCreateShader(GL_VERTEX_SHADER);
        shader = _vertex;
    } else {
        _fragment = glCreateShader(GL_FRAGMENT_SHADER);
        shader = _fragment;
    }

    glShaderSource(shader,1,(const GLchar**)&src,0);
    glCompileShader(shader);

    GLint isCompiled;
    glGetShaderiv(shader,GL_COMPILE_STATUS,&isCompiled);
    if(isCompiled == GL_FALSE){
        GLint maxLength;
        glGetShaderiv(shader,GL_INFO_LOG_LENGTH,&maxLength);
        char* infoLog = (char*)malloc(maxLength);
        glGetShaderInfoLog(shader,maxLength,&maxLength,infoLog);

        std::cout << infoLog << std::endl;
        free(infoLog);
        return false;
    }

    glAttachShader(_shader,shader);
    if(type == Vertex){
        _vertexAttached = true;
    }else{
        _fragmentAttached = true;
    }
    return true;
}

bool ShaderHandler::link() {

    glLinkProgram(_shader);

    GLint isLinked;
    glGetProgramiv(_shader,GL_LINK_STATUS,&isLinked);

    if(isLinked == GL_FALSE){
        GLint maxLength;
        glGetShaderiv(_shader,GL_INFO_LOG_LENGTH,&maxLength);
        char* infoLog = (char*)malloc(maxLength);
        glGetShaderInfoLog(_shader,maxLength,&maxLength,infoLog);
        std::cout << infoLog << std::endl;
        free(infoLog);
        return false;
    }
    return true;
}

void ShaderHandler::useProgram() const {

    glUseProgram(_shader);
}
void ShaderHandler::stopUsingProgram() const {

    glUseProgram(0);
}

//GLuint ShaderHandler::id(){return _shader;}

void ShaderHandler::setUniformi(const char* name,GLint value) {

    GLint location = glGetUniformLocation(_shader,(const GLchar*)name);
    if(location != -1) {
        glUniform1i(location,value);
    } else {
        printf("Location FAILED %s\n", name);
    }
}

void ShaderHandler::setUniformf(const char* name,GLfloat value) {

    GLint location = glGetUniformLocation(_shader,(const GLchar*)name);
    if(location != -1) {
        glUniform1f(location,value);
    } else {
        printf("Location FAILED %s\n", name);
    }
}

void ShaderHandler::setUniform2fv(const char* name,float value[2]) {

    GLint location = glGetUniformLocation(_shader,(const GLchar*)name);
    if(location != -1) {
        glUniform2fv(location,1,value);
    } else {
        printf("Location FAILED %s\n", name);
    }
}

void ShaderHandler::setUniform3fv(const char* name,float value[3]) {

    GLint location = glGetUniformLocation(_shader,(const GLchar*)name);
    if(location != -1) {
        glUniform3fv(location,1,value);
    } else {
        printf("Location FAILED %s\n", name);
    }
}

void ShaderHandler::setUniformArrayi(const char* name, int size, const GLint *textList) {

    GLint location = glGetUniformLocation(_shader, (const GLchar*)name);

    if(location != -1) {
        glUniform1iv(location, size, textList);
    } else {
        printf("Location FAILED %s\n", name);
    }
}

void ShaderHandler::setUniformArrayf(const char* name, int size, const GLfloat *values) {

    GLint location = glGetUniformLocation(_shader, (const GLchar*)name);

    if(location != -1) {
        glUniform1fv(location, size, values);
    } else {
        printf("Location FAILED %s\n", name);
    }
}

void ShaderHandler::setUniformMat3f(const char* name, GLfloat *mat) {

    GLint location = glGetUniformLocation(_shader, (const GLchar*)name);

    if(location != -1) {
        glUniformMatrix3fv(location, 1, 0, mat);
    } else {
        printf("Location FAILED %s\n", name);
    }
}

void ShaderHandler::setUniformMat4x3f(const char* name, GLfloat *mat) {

    GLint location = glGetUniformLocation(_shader, (const GLchar*)name);

    if(location != -1) {
        glUniformMatrix4x3fv(location, 1, 0, mat);
    } else {
        printf("Location FAILED %s\n", name);
    }
}

void ShaderHandler::setUniformMat4f(const char* name, GLfloat *mat) {

    GLint location = glGetUniformLocation(_shader, (const GLchar*)name);

    if(location != -1) {
        glUniformMatrix4fv(location, 1, 0, mat);
    } else {
        printf("Location FAILED %s\n", name);
    }
}

void ShaderHandler::bindAttribLocation(int pos, const char* name) {

    glBindAttribLocation(_shader, pos, name);
}

void startRenderingTo(GLuint fboId, GLenum attachment,int w, int h) {

    glBindFramebuffer(GL_FRAMEBUFFER,fboId);
    glDrawBuffer(attachment);
    glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT);
    glViewport(0,0,w,h);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0,w,0,h,-1,1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
}

void stopRenderingTo() {

    glPopAttrib();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glBindFramebuffer(GL_FRAMEBUFFER,0);
}

int loadshader(const char* filename, GLchar** ShaderSource, unsigned long &len) {

    len = 0;
    std::ifstream file;
    file.open(filename, std::ios::in); // opens as ASCII!
    if(!file) {
        return -1;
    }

    if (!file.good()) {
        return -2;   // Error: Empty File
    }

    {
        file.seekg(0,std::ios::end);
        len = file.tellg();
        file.seekg(std::ios::beg);
    }

    *ShaderSource = (GLchar*) new char[len+1];
    if (*ShaderSource == 0) {
        return -3;   // can't reserve memory
    }

    // len isn't always strlen cause some characters are stripped in ascii read...
    // it is important to 0-terminate the real length later, len is just max possible value...
    (*ShaderSource)[len] = 0;

    unsigned int i=0;

    while (file.good())
    {
        (*ShaderSource)[i] = file.get();       // get character from file.
        if (file.good())
            i++;
    }

    (*ShaderSource)[i] = 0;  // 0-terminate it at the correct position

    file.close();

    return 0; // No Error
}

int unloadshader(GLubyte** ShaderSource) {

    if (*ShaderSource != 0) {
        delete[] *ShaderSource;
    }
    *ShaderSource = 0;

    return 0; // No Error
}

