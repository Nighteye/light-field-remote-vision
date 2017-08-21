#version 410

in float vi_depth;
out vec4 out_depthTex;

void main() {

    out_depthTex = vec4(vi_depth, 0.0, 0.0, 1.0);
}
