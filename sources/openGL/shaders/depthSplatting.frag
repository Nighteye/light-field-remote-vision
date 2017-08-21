#version 410

in vec3 quadX;

uniform mat3 u_K;
uniform mat3 u_R;
uniform vec3 u_t;

out vec4 outZ;

void main() {

    vec3 xu = u_K*(u_R*quadX + u_t);

    outZ = vec4(xu.z, 0.0, 0.0, 1.0);
}
