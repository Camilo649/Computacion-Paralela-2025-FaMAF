#pragma once

#include <time.h> // time

#ifndef SHELLS
#define SHELLS 101 // discretization level
#endif

#ifndef PHOTONS_GPU
#define PHOTONS_GPU 6442450944UL // 64G photons
#endif

#ifndef PHOTONS_CPU
#define PHOTONS_CPU PHOTONS_GPU * 0.09375 // 9,375%
#endif

#ifndef MU_A
#define MU_A 2.0f // Absorption Coefficient in 1/cm !!non-zero!!
#endif

#ifndef MU_S
#define MU_S 20.0f // Reduced Scattering Coefficient in 1/cm
#endif

#ifndef MICRONS_PER_SHELL
#define MICRONS_PER_SHELL 50 // Thickness of spherical shells in microns
#endif

#ifndef SEED
#define SEED ((uint32_t)(time(NULL) & 0xFFFFFFFF)) // Random seed
#endif

#ifndef PHOTONS_PER_THREAD
#define PHOTONS_PER_THREAD 128UL
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 128UL
#endif

