#version 410

in vec2 textureCoord;

uniform sampler2DRect tex1;
uniform sampler2DRect tex2;

out vec4 outColor;

void main() {

    outColor = texture(tex1, textureCoord) + texture(tex2, textureCoord)/10;
}
