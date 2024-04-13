/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_EIG_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_EIG_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>

namespace EIG_BACKWARD
{
void eig_backward(
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* depth,
		const float* final_Ts,
		const uint32_t* n_contrib,
		// params for preprocess in backward
		int P, int D, int M,
		const float3* means3D,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* viewmatrix,
		const float* projmatrix,
		const float* projmatrix_raw,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const glm::vec3* dE_dmean3D,
		const float* dE_dopacity,
		const float* dE_dsh,
		const glm::vec3* dE_dscale,
		const glm::vec4* dE_drot,
		const float* dE_dtaus,
		// output
		float* dE_dpixels,
		float* dE_dpixels_depth,
		const int grad_power
	);
}

#endif