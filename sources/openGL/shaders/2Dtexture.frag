#version 410

in vec2 textureCoord;

uniform sampler2DRect myTexture;
uniform int H;

out vec4 outColor;

void main() {

    outColor = texture(myTexture, vec2(textureCoord.x, H - textureCoord.y));
}
