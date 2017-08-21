#version 410

in vec2 in_pixel;
in vec2 in_textureCoord;

out vec2 textureCoord;

void main() {

    gl_Position = vec4( in_pixel, 0.0, 1.0 );
    textureCoord = in_textureCoord;
}
