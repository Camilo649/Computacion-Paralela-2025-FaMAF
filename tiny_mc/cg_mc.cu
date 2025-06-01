#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "params.h"
#include "photon.cuh"
#include "xorshift32.cuh"

#define PHOTON_CAP 1 << 16
#define MAX_PHOTONS_PER_FRAME 20
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR %s:%d: %s\n", file, line, cudaGetErrorString(code));
        exit(code);
    }
}
#define GL_CHECK() { glCheck(__FILE__, __LINE__); }
inline void glCheck(const char* file, int line)
{
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL ERROR %s:%d: 0x%X\n", file, line, err);
        exit(1);
    }
}


static const char* VSHADER = R"(
#version 430
vec2 vertices[4] = vec2[](
    vec2(-1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0)
);
uint indices[6] = uint[](0, 1, 2, 0, 2, 3);
void main() {
    gl_Position = vec4(vertices[indices[gl_VertexID]], 0.0, 1.0);
})";

static const char* FSHADER = R"(
#version 430
#define MC 0.7071067811865476f
out vec4 frag_color;
layout(std430, binding = 0) readonly buffer ssbo {
    float heats[];
} shells;
void main() {
    vec2 uv = gl_FragCoord.xy / vec2(800);
    float dr = length(uv - vec2(0.5));
    int heat_id = int((dr / MC) * float(shells.heats.length() - 1));
    float heat = shells.heats[heat_id];
    float L = 2.0, b = 1.0, k = 0.004;
    float heat_fit = L / (1.0 + b * exp(-k * heat)) - 1.0;
    frag_color = vec4(heat_fit, 0.0, 0.0, 1.0);
})";

__global__ void simulate_kernel(float* heats, float* heats_squared)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= PHOTONS) return;
    Xorshift32 rng;
    xorshift32_init(&rng, SEED ^ (threadIdx.x * 0x9E3779B9u) ^ (blockIdx.x * 0x85EBCA6Bu) ^ (blockDim.x * 0xC2B2AE35u));
    photon(heats, heats_squared, &rng);
}

int main(void) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(800, 800, "tiny mc", NULL, NULL);
    assert(window);
    glfwMakeContextCurrent(window);
   
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "GLEW initialization failed\n");
        return -1;
    }


    GLuint vshader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vshader, 1, &VSHADER, NULL);
    glCompileShader(vshader);
    GL_CHECK();
    GLint status;
    glGetShaderiv(vshader, GL_COMPILE_STATUS, &status);
    assert(status);

    GLuint fshader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fshader, 1, &FSHADER, NULL);
    glCompileShader(fshader);
    GL_CHECK();
    glGetShaderiv(fshader, GL_COMPILE_STATUS, &status);
    assert(status);

    GLuint program = glCreateProgram();
    glAttachShader(program, vshader);
    glAttachShader(program, fshader);
    glLinkProgram(program);
    GL_CHECK();
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    assert(status);

    glDeleteShader(vshader);
    glDeleteShader(fshader);
    glUseProgram(program);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glViewport(0, 0, 800, 800);

    GLuint ssbo_heats, ssbo_heats_squared;
    cudaGraphicsResource* cuda_res_heats;
    cudaGraphicsResource* cuda_res_heats_squared;

    glGenBuffers(1, &ssbo_heats);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_heats);
    glBufferData(GL_SHADER_STORAGE_BUFFER, SHELLS * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_heats);
    glFinish();

    glGenBuffers(1, &ssbo_heats_squared);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_heats_squared);
    glBufferData(GL_SHADER_STORAGE_BUFFER, SHELLS * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_heats_squared);
    glFinish();

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_res_heats, ssbo_heats, cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_res_heats_squared, ssbo_heats_squared, cudaGraphicsMapFlagsWriteDiscard));

    glfwShowWindow(window);

    int remaining_photons = PHOTON_CAP;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (remaining_photons > 0) {
	    int photons_this_frame = min(MAX_PHOTONS_PER_FRAME, remaining_photons);
            remaining_photons -= photons_this_frame;
		
	    glFinish();

            cudaGraphicsResource* resources[] = { cuda_res_heats, cuda_res_heats_squared };
	    CUDA_CHECK(cudaGraphicsMapResources(2, resources));

	    float* dev_heats;
	    float* dev_heats_squared;
	    size_t size;

	    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dev_heats, &size, cuda_res_heats));
            CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dev_heats_squared, &size, cuda_res_heats_squared));

	    simulate_kernel<<<(photons_this_frame + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dev_heats, dev_heats_squared);
	    cudaDeviceSynchronize();

	    CUDA_CHECK(cudaGraphicsUnmapResources(2, resources));
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glfwSwapBuffers(window);
    }

    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_res_heats));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_res_heats_squared));

    glDeleteBuffers(1, &ssbo_heats);
    glDeleteBuffers(1, &ssbo_heats_squared);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
