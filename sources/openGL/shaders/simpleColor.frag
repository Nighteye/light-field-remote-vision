// Version du GLSL

#version 150 core


// in

in vec3 color;

// out

out vec4 out_Color;

void main() {

    // Final pixel color

    out_Color = vec4(color, 1.0);
}
