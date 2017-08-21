// Version du GLSL

#version 150 core

in vec3 in_Vertex;
in vec3 in_Color;

uniform mat4 modelviewProjection;

out vec3 color;

void main() {

    gl_Position = modelviewProjection * vec4(in_Vertex, 1.0);

    color = in_Color;
}
