#version 410

in vec2 textureCoord;

uniform sampler2DRect lowScaleImage;

uniform int horizontal;

out vec4 color;

void main() {

    float a = 0.4;

    float w[5];
    w[0] = 0.25 - 0.5*a;
    w[1] = 0.25;
    w[2] = a;
    w[3] = w[0];
    w[4] = w[1];
    vec3 in0, in1, in2, in3, in4;

    if(horizontal == 1) {

        in0 = texture( lowScaleImage, vec2(2*textureCoord.x - 2, textureCoord.y) ).xyz;
        in1 = texture( lowScaleImage, vec2(2*textureCoord.x - 1, textureCoord.y) ).xyz;
        in2 = texture( lowScaleImage, vec2(2*textureCoord.x, textureCoord.y) ).xyz;
        in3 = texture( lowScaleImage, vec2(2*textureCoord.x + 1, textureCoord.y) ).xyz;
        in4 = texture( lowScaleImage, vec2(2*textureCoord.x + 2, textureCoord.y) ).xyz;

    } else {

        in0 = texture( lowScaleImage, vec2(textureCoord.x, 2*textureCoord.y - 2) ).xyz;
        in1 = texture( lowScaleImage, vec2(textureCoord.x, 2*textureCoord.y - 1) ).xyz;
        in2 = texture( lowScaleImage, vec2(textureCoord.x, 2*textureCoord.y) ).xyz;
        in3 = texture( lowScaleImage, vec2(textureCoord.x, 2*textureCoord.y + 1) ).xyz;
        in4 = texture( lowScaleImage, vec2(textureCoord.x, 2*textureCoord.y + 2) ).xyz;
    }

    color = vec4(in0*w[0] + in1*w[1] + in2*w[2] + in3*w[3] + in4*w[4], 1.0);
}
