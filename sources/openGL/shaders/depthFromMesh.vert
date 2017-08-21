#version 410

in vec3 in_vertex;

uniform mat4 renderMatrix;
uniform mat3 vk_K;
uniform mat3 vk_R;
uniform vec3 vk_t;
uniform float offset;

out float vi_depth;

void main() {

    // Compute depth as seen from vi
    vi_depth = vec3(vk_K*(vk_R*in_vertex+vk_t)).z;

    gl_Position = renderMatrix * vec4(in_vertex, 1.) + offset;
}
