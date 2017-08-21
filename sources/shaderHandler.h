//This Source Code Form is subject to the terms of the Mozilla Public
//  License, v. 2.0. If a copy of the MPL was not distributed with this
//  file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Contact: immarespond at gmail dot com
// 2013, Alexandre Gauthier

#ifndef SHADER_HANDLER_H
#define SHADER_HANDLER_H

#include <GL/glew.h>

enum ShaderType{Vertex,Fragment};

class ShaderHandler
{
   GLuint _shader;
   GLuint _vertex;
   GLuint _fragment;
   bool _vertexAttached,_fragmentAttached;

public:
    ShaderHandler();

    ~ShaderHandler();

    bool addShader(ShaderType type,const char* src);

    bool link();

    void useProgram() const;
    void stopUsingProgram() const;

    GLuint id(){return _shader;}

    void setUniformi(const char* name,GLint value);
    void setUniformf(const char* name,GLfloat value);
    void setUniform2fv(const char* name,float value[2]);
    void setUniform3fv(const char* name,float value[3]);
    void setUniformArrayi(const char* name, int size, const GLint *textList);
    void setUniformArrayf(const char* name, int size, const GLfloat *values);
    void setUniformMat3f(const char* name, GLfloat *mat);
    void setUniformMat4x3f(const char* name, GLfloat *mat);
    void setUniformMat4f(const char* name, GLfloat *mat);
    //void setAttributeArray(const char* name, const float *values);

    void bindAttribLocation(int pos, const char* name);
    //void bindFragDataLocation(int pos, const char* name);

};

void startRenderingTo(GLuint fboId, GLenum attachment,int w, int h);
void stopRenderingTo();

int loadshader(const char* filename, GLchar** ShaderSource, unsigned long &len);

int unloadshader(GLubyte** ShaderSource);

#endif // SHADER_HANDLER_H
