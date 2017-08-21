#version 410

in vec2 in_pixel;
in vec2 in_textureCoord;

void main() {

    gl_Position = vec4(in_pixel.x, in_pixel.y, 0.0, 1.0 );
}
