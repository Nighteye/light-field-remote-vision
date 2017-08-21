#version 410

in vec3 in_vertex;
in vec3 in_normal;

out vec3 gnormal;

void main() {

    gnormal = in_normal;

    gl_Position = vec4(in_vertex, 1.);
}
