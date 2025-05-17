#pragma once

#ifndef SHELLS
#define SHELLS 101 // discretization level
#endif

#ifndef PHOTONS
#define PHOTONS 17179869184UL // 32G photons
#endif

#ifndef THREADS
#define THREADS 48UL
#endif

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 16UL
#endif

#ifndef MU_A
#define MU_A 2.0f // Absorption Coefficient in 1/cm !!non-zero!!
#endif

#ifndef MU_S
#define MU_S 20.0f // Reduced Scattering Coefficient in 1/cm
#endif

#ifndef MICRONS_PER_SHELL
#define MICRONS_PER_SHELL 25 // Thickness of spherical shells in microns
#endif

#ifndef SEED
#define SEED (time(NULL)) // random seed
#endif


