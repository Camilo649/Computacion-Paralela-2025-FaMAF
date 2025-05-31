#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "params.h"
#include "photon.cuh"
#include "xorshift32.cuh"

#define PHOTON_CAP 1 << 16
#define MAX_PHOTONS_PER_FRAME 20

static float heats[SHELLS];
static float _heats_squared[SHELLS];
static int remaining_photons = PHOTON_CAP;

static cudaGraphicsResource* cuda_heats_resource;
static cudaGraphicsResource* cuda_heats_squared_resource;
static GLuint ssbo;
static GLuint ssbo_squared;

__global__ void simulate_kernel(float* heats, float* heats_squared)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= PHOTONS) return;
    Xorshift32 rng;
    xorshift32_init(&rng, (SEED ^ tid) + 1);  // semilla única por hilo
    photon(heats, heats_squared, &rng);
}

void launch_photon_simulation(float* heats_dev, float* heats_squared_dev, int photons_this_frame) {
    int blocks = (photons_this_frame + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    photon_kernel<<<blocks, THREADS_PER_BLOCK>>>(heats_dev, heats_squared_dev, photons_this_frame);
    cudaDeviceSynchronize();
}

// clang-format off
static const char *VSHADER = ""
"#version 430\n"
"vec2 vertices[4] = {\n"
"    {-1.0,  1.0},\n"
"    {-1.0, -1.0},\n"
"    { 1.0, -1.0},\n"
"    { 1.0,  1.0}\n"
"};\n"
"uint indices[6] = {0, 1, 2, 0, 2, 3};\n"
"void main() {\n"
"    gl_Position = vec4(vertices[indices[gl_VertexID]], 0.0, 1.0);\n"
"}";

static const char *FSHADER = ""
"#version 430\n"
"#define MC 0.7071067811865476f\n"
"out vec4 frag_color;\n"
"layout(std430, binding = 0) readonly buffer ssbo {\n"
"    float heats[];\n"
"} shells;\n"
"void main() {\n"
"    vec2 uv = gl_FragCoord.xy / vec2(800);\n"
"    float dr = length(uv - vec2(0.5));\n"
"    int heat_id = int((dr / MC) * float(shells.heats.length() - 1));\n"
"    float heat = shells.heats[heat_id];\n"
"    float L = 2.0;\n"
"    float b = 1.0;\n"
"    float k = 0.004;\n"
"    float heat_fit = L / (1.0 + b * exp(-k * heat)) - 1.0;\n"
"    frag_color = vec4(heat_fit, 0.0, 0.0, 1.0);\n"
"}";
// clang-format on

void update() {
    if (remaining_photons <= 0) return;

    int photons_this_frame = MAX_PHOTONS_PER_FRAME;
    if (photons_this_frame > remaining_photons)
        photons_this_frame = remaining_photons;

    float* d_heats = NULL;
    float* d_heats_squared = NULL;
    size_t bytes;

    cudaGraphicsMapResources(1, &cuda_heats_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_heats, &bytes, cuda_heats_resource);

    cudaGraphicsMapResources(1, &cuda_heats_squared_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_heats_squared, &bytes, cuda_heats_squared_resource);

    launch_photon_simulation(d_heats, d_heats_squared, photons_this_frame);

    cudaGraphicsUnmapResources(1, &cuda_heats_resource, 0);
    cudaGraphicsUnmapResources(1, &cuda_heats_squared_resource, 0);

    remaining_photons -= photons_this_frame;
}

int main(void) {
    glfwInit();

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(800, 800, "tiny mc", NULL, NULL);
    assert(window != NULL);

    glfwMakeContextCurrent(window);
    glewInit();

    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &VSHADER, NULL);
    glCompileShader(vertex_shader);

    GLint status;
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &status);
    assert(status);

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &FSHADER, NULL);
    glCompileShader(fragment_shader);

    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &status);
    assert(status);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &status);
    assert(status);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    glUseProgram(program);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glViewport(0, 0, 800, 800);
    glClearColor(0.0, 0.0, 0.0, 1.0);

    glfwShowWindow(window);

    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(heats), heats, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &ssbo_squared);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_squared);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_squared);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(_heats_squared), _heats_squared, GL_DYNAMIC_DRAW);

    cudaGraphicsGLRegisterBuffer(&cuda_heats_resource, ssbo, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_heats_squared_resource, ssbo_squared, cudaGraphicsMapFlagsWriteDiscard);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        update();

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
    }

    cudaGraphicsUnregisterResource(cuda_heats_resource);
    cudaGraphicsUnregisterResource(cuda_heats_squared_resource);

    glDeleteBuffers(1, &ssbo);
    glDeleteBuffers(1, &ssbo_squared);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
}
