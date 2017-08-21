#version 410

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec3 gnormal[];

uniform mat4 renderMatrix;

out vec3 currentPos;
out vec3 fnormal;

void main() {

    currentPos = gl_in[0].gl_Position.xyz;

    float offset = 0.05;
    vec3 N = gnormal[0];
    N = N / sqrt(dot(N, N));
    fnormal = N;

    // in radian
    float phi, theta;
    phi = asin(N.y);
    if(cos(phi) == 0.0){
        theta = 0.0;
    } else {
        theta = acos(N.z / cos(phi));
    }

    if(N.z < 0) {
        theta *= -1;
    }

    vec4 Rx1 = vec4(1.0, 0.0, 0.0, 0.0);
    vec4 Rx2 = vec4(0.0, cos(phi), sin(phi), 0.0);
    vec4 Rx3 = vec4(0.0, -sin(phi), cos(phi), 0.0);
    vec4 Rx4 = vec4(0.0, 0.0, 0.0, 1.0);

    vec4 Ry1 = vec4(cos(theta), 0.0, -sin(theta), 0.0);
    vec4 Ry2 = vec4(0.0, 1.0, 0.0, 0.0);
    vec4 Ry3 = vec4(sin(theta), 0.0, cos(theta), 0.0);
    vec4 Ry4 = Rx4;

    mat4 Rx, Ry, R;
    Rx[0] = Rx1; Rx[1] = Rx2; Rx[2] = Rx3; Rx[3] = Rx4;
    Ry[0] = Ry1; Ry[1] = Ry2; Ry[2] = Ry3; Ry[3] = Ry4;
    R = Rx*Ry;

    gl_Position = renderMatrix * (gl_in[0].gl_Position + R*vec4( -offset, -offset,  0.0, 0.0));
    EmitVertex();
    gl_Position = renderMatrix * (gl_in[0].gl_Position + R*vec4( -offset, offset, 0.0, 0.0));
    EmitVertex();
    gl_Position = renderMatrix * (gl_in[0].gl_Position + R*vec4( offset, -offset, 0.0, 0.0));
    EmitVertex();
    gl_Position = renderMatrix * (gl_in[0].gl_Position + R*vec4( offset, offset, 0.0, 0.0));
    EmitVertex();

    EndPrimitive();
}
