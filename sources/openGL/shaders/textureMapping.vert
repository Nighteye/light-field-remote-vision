#version 410

uniform mat4 renderMatrix;

in vec3 in_position;

uniform mat4x3 vi_P;
uniform mat3 vi_R;
uniform vec3 vi_C;

out vec3 vi_coordinate;

void main() {

    gl_Position = renderMatrix * vec4( in_position, 1. );

    // Project vertex into vi to know which texture coordinate to use
    vec3 vi_PX = vi_P * vec4(in_position, 1.);
    float vi_depth = vec3( vi_R * ( in_position - vi_C ) ).z;

    vi_coordinate = vec3( vi_PX.xy/vi_PX.z * vi_depth, vi_depth );
}
