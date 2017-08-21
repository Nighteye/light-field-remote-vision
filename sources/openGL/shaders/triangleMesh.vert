#version 150 core

in vec3 in_position;
//in vec3 in_normal;

uniform vec3 in_color;
uniform mat4 modelviewProjection;

out vec3 color;

void main() {

    gl_Position = modelviewProjection * vec4(in_position, 1.0);

    color = in_color;
}
