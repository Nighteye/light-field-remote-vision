#version 410

in vec4 fcolor;

out vec4 splattedColor;

void main() {

    splattedColor = fcolor;
    // splattedColor = vec4(gl_FragCoord.z, gl_FragCoord.z, gl_FragCoord.z, gl_FragCoord.z);
}
