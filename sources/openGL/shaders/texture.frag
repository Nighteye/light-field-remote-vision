// Version du GLSL

#version 150 core

in vec2 coordTexture;

uniform sampler2DRect myTexture;
uniform float alpha;
uniform int H;

out vec4 out_Color;

void main() {

    out_Color = vec4( texture( myTexture, vec2(coordTexture.x, H - coordTexture.y) ).xyz, alpha );
}
