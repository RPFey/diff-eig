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

#include "fisher_backward.h"
#include "auxiliary.h"
#include "math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
// backpropogate dE_dshs, dE_dmeans, dE_dtau to dE_dcolor
__device__ void computeColorFromSH_EIG(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs,  float *dL_dtau)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);

	dL_dtau[6 * idx + 0] += -dL_dmean.x;
	dL_dtau[6 * idx + 1] += -dL_dmean.y;
	dL_dtau[6 * idx + 2] += -dL_dmean.z;

}

__device__ void computeColorFromSH_backward(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor,
	const glm::vec3* dE_dmeans, const float* dE_dshs, const float *dE_dtaus, float* dE_dcolors)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	// glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	// float dRGBdsh0 = SH_C0;
	// dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		// float dRGBdsh1 = -SH_C1 * y;
		// float dRGBdsh2 = SH_C1 * z;
		// float dRGBdsh3 = -SH_C1 * x;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	glm::vec3 dE_dmean(-dE_dtaus[6 * idx + 0], -dE_dtaus[6 * idx + 1], -dE_dtaus[6 * idx + 2]);
	dE_dmean += dE_dmeans[idx];

	float3 dE_ddir = dnormvdv_backward(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z }, float3{dE_dmean.x, dE_dmean.y, dE_dmean.z});

	glm::vec3 dE_dRGB(0.f);
	dE_dRGB.x = dRGBdx.x * dE_ddir.x + dRGBdy.x * dE_ddir.y + dRGBdz.x * dE_ddir.z;
	dE_dRGB.y = dRGBdx.y * dE_ddir.x + dRGBdy.y * dE_ddir.y + dRGBdz.y * dE_ddir.z;
	dE_dRGB.z = dRGBdx.z * dE_ddir.x + dRGBdy.z * dE_ddir.y + dRGBdz.z * dE_ddir.z;

 	// float dRGBdsh0 = SH_C0;
	const float* dE_dsh = dE_dshs + idx * max_coeffs;
	dE_dRGB += dE_dsh[0] * SH_C0;
	if (deg > 0){
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;

		dE_dRGB += dE_dsh[1] * dRGBdsh1;
		dE_dRGB += dE_dsh[2] * dRGBdsh2;
		dE_dRGB += dE_dsh[3] * dRGBdsh3;
		//!TODO: implement for higher degree as well
	}

	float* dE_dcolor = dE_dcolors + idx;
	dE_dcolor[0] += clamped[3 * idx + 0] ? 0 : dE_dRGB.x;
	dE_dcolor[1] += clamped[3 * idx + 1] ? 0 : dE_dRGB.y;
	dE_dcolor[2] += clamped[3 * idx + 2] ? 0 : dE_dRGB.z;

}

//!NOTE: not used in the backward of EIG
// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
// __global__ void computeCov2DCUDA(int P,
// 	const float3* means,
// 	const int* radii,
// 	const float* cov3Ds,
// 	const float h_x, float h_y,
// 	const float tan_fovx, float tan_fovy,
// 	const float* view_matrix,
// 	const float* dL_dconics,
// 	float3* dL_dmeans,
// 	float* dL_dcov,
// 	float *dL_dtau)
// {
// 	auto idx = cg::this_grid().thread_rank();
// 	if (idx >= P || !(radii[idx] > 0))
// 		return;

// 	// Reading location of 3D covariance for this Gaussian
// 	const float* cov3D = cov3Ds + 6 * idx;

// 	// Fetch gradients, recompute 2D covariance and relevant 
// 	// intermediate forward results needed in the backward.
// 	float3 mean = means[idx];
// 	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
// 	float3 t = transformPoint4x3(mean, view_matrix);
	
// 	const float limx = 1.3f * tan_fovx;
// 	const float limy = 1.3f * tan_fovy;
// 	const float txtz = t.x / t.z;
// 	const float tytz = t.y / t.z;
// 	t.x = min(limx, max(-limx, txtz)) * t.z;
// 	t.y = min(limy, max(-limy, tytz)) * t.z;
	
// 	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
// 	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

// 	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
// 		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
// 		0, 0, 0);

// 	glm::mat3 W = glm::mat3(
// 		view_matrix[0], view_matrix[4], view_matrix[8],
// 		view_matrix[1], view_matrix[5], view_matrix[9],
// 		view_matrix[2], view_matrix[6], view_matrix[10]);

// 	glm::mat3 Vrk = glm::mat3(
// 		cov3D[0], cov3D[1], cov3D[2],
// 		cov3D[1], cov3D[3], cov3D[4],
// 		cov3D[2], cov3D[4], cov3D[5]);

// 	glm::mat3 T = W * J;

// 	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

// 	// Use helper variables for 2D covariance entries. More compact.
// 	float a = cov2D[0][0] += 0.3f;
// 	float b = cov2D[0][1];
// 	float c = cov2D[1][1] += 0.3f;

// 	float denom = a * c - b * b;
// 	float dL_da = 0, dL_db = 0, dL_dc = 0;
// 	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

// 	if (denom2inv != 0)
// 	{
// 		// Gradients of loss w.r.t. entries of 2D covariance matrix,
// 		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
// 		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
// 		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
// 		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
// 		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

// 		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
// 		// given gradients w.r.t. 2D covariance matrix (diagonal).
// 		// cov2D = transpose(T) * transpose(Vrk) * T;
// 		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
// 		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
// 		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

// 		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
// 		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
// 		// Off-diagonal elements appear twice --> double the gradient.
// 		// cov2D = transpose(T) * transpose(Vrk) * T;
// 		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
// 		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
// 		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
// 	}
// 	else
// 	{
// 		for (int i = 0; i < 6; i++)
// 			dL_dcov[6 * idx + i] = 0;
// 	}

// 	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
// 	// cov2D = transpose(T) * transpose(Vrk) * T;
// 	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
// 		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
// 	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
// 		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
// 	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
// 		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
// 	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
// 		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
// 	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
// 		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
// 	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
// 		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

// 	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
// 	// T = W * J
// 	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
// 	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
// 	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
// 	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

// 	float tz = 1.f / t.z;
// 	float tz2 = tz * tz;
// 	float tz3 = tz2 * tz;

// 	// Gradients of loss w.r.t. transformed Gaussian mean t
// 	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
// 	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
// 	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

// 	SE3 T_CW(view_matrix);
// 	mat33 R = T_CW.R().data();
// 	mat33 RT = R.transpose();
// 	float3 t_ = T_CW.t();
// 	mat33 dpC_drho = mat33::identity();
// 	mat33 dpC_dtheta = -mat33::skew_symmetric(t);
// 	float dL_dt[6];
// 	for (int i = 0; i < 3; i++) {
// 		float3 c_rho = dpC_drho.cols[i];
// 		float3 c_theta = dpC_dtheta.cols[i];
// 		dL_dt[i] = dL_dtx * c_rho.x + dL_dty * c_rho.y + dL_dtz * c_rho.z;
// 		dL_dt[i + 3] = dL_dtx * c_theta.x + dL_dty * c_theta.y + dL_dtz * c_theta.z;
// 	}
// 	for (int i = 0; i < 6; i++) {
// 		dL_dtau[6 * idx + i] += dL_dt[i];
// 	}

// 	// Account for transformation of mean to t
// 	// t = transformPoint4x3(mean, view_matrix);
// 	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

// 	// Gradients of loss w.r.t. Gaussian means, but only the portion 
// 	// that is caused because the mean affects the covariance matrix.
// 	// Additional mean gradient is accumulated in BACKWARD::preprocess.
// 	dL_dmeans[idx] = dL_dmean;

// 	float dL_dW00 = J[0][0] * dL_dT00;
// 	float dL_dW01 = J[0][0] * dL_dT01;
// 	float dL_dW02 = J[0][0] * dL_dT02;
// 	float dL_dW10 = J[1][1] * dL_dT10;
// 	float dL_dW11 = J[1][1] * dL_dT11;
// 	float dL_dW12 = J[1][1] * dL_dT12;
// 	float dL_dW20 = J[0][2] * dL_dT00 + J[1][2] * dL_dT10;
// 	float dL_dW21 = J[0][2] * dL_dT01 + J[1][2] * dL_dT11;
// 	float dL_dW22 = J[0][2] * dL_dT02 + J[1][2] * dL_dT12;

// 	float3 c1 = R.cols[0];
// 	float3 c2 = R.cols[1];
// 	float3 c3 = R.cols[2];

// 	float dL_dW_data[9];
// 	dL_dW_data[0] = dL_dW00;
// 	dL_dW_data[3] = dL_dW01;
// 	dL_dW_data[6] = dL_dW02;
// 	dL_dW_data[1] = dL_dW10;
// 	dL_dW_data[4] = dL_dW11;
// 	dL_dW_data[7] = dL_dW12;
// 	dL_dW_data[2] = dL_dW20;
// 	dL_dW_data[5] = dL_dW21;
// 	dL_dW_data[8] = dL_dW22;

// 	mat33 dL_dW(dL_dW_data);
// 	float3 dL_dWc1 = dL_dW.cols[0];
// 	float3 dL_dWc2 = dL_dW.cols[1];
// 	float3 dL_dWc3 = dL_dW.cols[2];

// 	mat33 n_W1_x = -mat33::skew_symmetric(c1);
// 	mat33 n_W2_x = -mat33::skew_symmetric(c2);
// 	mat33 n_W3_x = -mat33::skew_symmetric(c3);

// 	float3 dL_dtheta = {};
// 	dL_dtheta.x = dot(dL_dWc1, n_W1_x.cols[0]) + dot(dL_dWc2, n_W2_x.cols[0]) +
// 				dot(dL_dWc3, n_W3_x.cols[0]);
// 	dL_dtheta.y = dot(dL_dWc1, n_W1_x.cols[1]) + dot(dL_dWc2, n_W2_x.cols[1]) +
// 				dot(dL_dWc3, n_W3_x.cols[1]);
// 	dL_dtheta.z = dot(dL_dWc1, n_W1_x.cols[2]) + dot(dL_dWc2, n_W2_x.cols[2]) +
// 				dot(dL_dWc3, n_W3_x.cols[2]);

// 	dL_dtau[6 * idx + 3] += dL_dtheta.x;
// 	dL_dtau[6 * idx + 4] += dL_dtheta.y;
// 	dL_dtau[6 * idx + 5] += dL_dtheta.z;


// }

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D_EIG(const int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	// float4* dL_drot = (float4*)(dL_drots + idx);
	glm::vec4* dL_drot = dL_drots + idx;
#if 1
	// float4 dL_drot;
	// dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
	// *dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
	dL_drot->x = dL_dq.x;
	dL_drot->y = dL_dq.y;
	dL_drot->z = dL_dq.z;
	dL_drot->w = dL_dq.w;
#endif
}


// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
// dE_drot, dE_dscale -> dE_dcov3D
__device__ void computeCov3D_EIG_backward(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, 
	const glm::vec4 dE_drot, const glm::vec3 dE_dscale, float* dE_dcov3Ds)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	// glm::vec3 dL_dscale = glm::vec3(0.f);
	// dL_dscale.x = glm::dot(Rt[0], dL_dMt[0]);
	// dL_dscale.y = glm::dot(Rt[1], dL_dMt[1]);
	// dL_dscale.z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	// Not needed in dE_drot
	// float4 dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });

	// Start back propogation
	const glm::vec4 dE_dq = dE_drot;
	glm::mat3 dE_dMt = glm::mat3(0.0f);	
	// dL_dq.x: 
	dE_dMt[0][1] += 2 * z * dE_dq.x;
	dE_dMt[1][0] += -2 * z * dE_dq.x;
	dE_dMt[0][2] += 2 * y * dE_dq.x;
	dE_dMt[2][0] += -2 * y * dE_dq.x;
	dE_dMt[1][2] += 2 * x * dE_dq.x;
	dE_dMt[2][1] += -2 * x * dE_dq.x;

	// dL_dq.y:
	dE_dMt[1][0] += 2 * y * dE_dq.y;
	dE_dMt[0][1] += 2 * y * dE_dq.y;
	dE_dMt[2][0] += 2 * z * dE_dq.y;
	dE_dMt[0][2] += 2 * z * dE_dq.y;
	dE_dMt[1][2] += 2 * r * dE_dq.y;
	dE_dMt[2][1] += -2 * r * dE_dq.y;
	dE_dMt[2][2] += -4 * x * dE_dq.y;
	dE_dMt[1][1] += -4 * x * dE_dq.y;

	// dL_dq.z:
	dE_dMt[1][0] += 2 * x * dE_dq.z;
	dE_dMt[0][1] += 2 * x * dE_dq.z;
	dE_dMt[2][0] += 2 * r * dE_dq.z;
	dE_dMt[0][2] += -2 * r * dE_dq.z;
	dE_dMt[1][2] += 2 * z * dE_dq.z;
	dE_dMt[2][1] += 2 * z * dE_dq.z;
	dE_dMt[2][2] += -4 * y * dE_dq.z;
	dE_dMt[0][0] += -4 * y * dE_dq.z;

	// dL_dq.w:
	dE_dMt[0][1] += 2 * r * dE_dq.w;
	dE_dMt[1][0] += -2 * r * dE_dq.w;
	dE_dMt[2][0] += 2 * x * dE_dq.w;
	dE_dMt[0][2] += 2 * x * dE_dq.w;
	dE_dMt[1][2] += 2 * y * dE_dq.w;
	dE_dMt[2][1] += 2 * y * dE_dq.w;
	dE_dMt[1][1] += -4 * z * dE_dq.w;
	dE_dMt[0][0] += -4 * z * dE_dq.w;

	dE_dMt[0] *= s.x;
	dE_dMt[1] *= s.y;
	dE_dMt[2] *= s.z;

	dE_dMt[0] += Rt[0] * dE_dscale.x;
	dE_dMt[1] += Rt[1] * dE_dscale.y;
	dE_dMt[2] += Rt[2] * dE_dscale.z;

	glm::mat3 dE_dM = glm::transpose(dE_dMt);

	glm::mat3 dE_dSgima = 2.0f * glm::transpose(M) * dE_dM;

	// NOTE: glm is column major
	float* dE_dcov3D = dE_dcov3Ds + 6 * idx;
	dE_dcov3D[0] += dE_dSgima[0][0];
	dE_dcov3D[1] += 0.5f * dE_dSgima[0][1] + 0.5f * dL_dSigma[1][0];
	dE_dcov3D[2] += 0.5f * dE_dSgima[0][2] + 0.5f * dL_dSigma[2][0];
	dE_dcov3D[3] += dE_dSgima[1][1];
	dE_dcov3D[4] += 0.5f * dE_dSgima[1][2] + 0.5f * dL_dSigma[2][1];
	dE_dcov3D[5] += dE_dSgima[2][2];
}


// template <typename T>
// __device__ void inline reduce_helper(int lane, int i, T *data) {
//   if (lane < i) {
//     data[lane] += data[lane + i];
//   }
// }

// template <typename group_t, typename... Lists>
// __device__ void render_cuda_reduce_sum(group_t g, Lists... lists) {
//   int lane = g.thread_rank();
//   g.sync();

//   for (int i = g.size() / 2; i > 0; i /= 2) {
//     (...,
//      reduce_helper(
//          lane, i, lists)); // Fold expression: apply reduce_helper for each list
//     g.sync();
//   }
// }

__device__ void computeCov2DCUDARelocated_EIG(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov,
	float* dL_dtau)
{
	// auto idx = cg::this_grid().thread_rank();
	// just the first since we've already shifted the pointer.
	int idx = 0;
	if (idx >= P || !(radii[idx] > 0))
		return;


	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	SE3 T_CW(view_matrix);
	mat33 R = T_CW.R().data();
	mat33 RT = R.transpose();
	float3 t_ = T_CW.t();
	mat33 dpC_drho = mat33::identity();
	mat33 dpC_dtheta = -mat33::skew_symmetric(t);
	float dL_dt[6];
	for (int i = 0; i < 3; i++) {
		float3 c_rho = dpC_drho.cols[i];
		float3 c_theta = dpC_dtheta.cols[i];
		dL_dt[i] = dL_dtx * c_rho.x + dL_dty * c_rho.y + dL_dtz * c_rho.z;
		dL_dt[i + 3] = dL_dtx * c_theta.x + dL_dty * c_theta.y + dL_dtz * c_theta.z;
	}
	for (int i = 0; i < 6; i++) {
		dL_dtau[6 * idx + i] += dL_dt[i];
	}

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;

	float dL_dW00 = J[0][0] * dL_dT00;
	float dL_dW01 = J[0][0] * dL_dT01;
	float dL_dW02 = J[0][0] * dL_dT02;
	float dL_dW10 = J[1][1] * dL_dT10;
	float dL_dW11 = J[1][1] * dL_dT11;
	float dL_dW12 = J[1][1] * dL_dT12;
	float dL_dW20 = J[0][2] * dL_dT00 + J[1][2] * dL_dT10;
	float dL_dW21 = J[0][2] * dL_dT01 + J[1][2] * dL_dT11;
	float dL_dW22 = J[0][2] * dL_dT02 + J[1][2] * dL_dT12;

	float3 c1 = R.cols[0];
	float3 c2 = R.cols[1];
	float3 c3 = R.cols[2];

	float dL_dW_data[9];
	dL_dW_data[0] = dL_dW00;
	dL_dW_data[3] = dL_dW01;
	dL_dW_data[6] = dL_dW02;
	dL_dW_data[1] = dL_dW10;
	dL_dW_data[4] = dL_dW11;
	dL_dW_data[7] = dL_dW12;
	dL_dW_data[2] = dL_dW20;
	dL_dW_data[5] = dL_dW21;
	dL_dW_data[8] = dL_dW22;

	mat33 dL_dW(dL_dW_data);
	float3 dL_dWc1 = dL_dW.cols[0];
	float3 dL_dWc2 = dL_dW.cols[1];
	float3 dL_dWc3 = dL_dW.cols[2];

	mat33 n_W1_x = -mat33::skew_symmetric(c1);
	mat33 n_W2_x = -mat33::skew_symmetric(c2);
	mat33 n_W3_x = -mat33::skew_symmetric(c3);

	float3 dL_dtheta = {};
	dL_dtheta.x = dot(dL_dWc1, n_W1_x.cols[0]) + dot(dL_dWc2, n_W2_x.cols[0]) +
				dot(dL_dWc3, n_W3_x.cols[0]);
	dL_dtheta.y = dot(dL_dWc1, n_W1_x.cols[1]) + dot(dL_dWc2, n_W2_x.cols[1]) +
				dot(dL_dWc3, n_W3_x.cols[1]);
	dL_dtheta.z = dot(dL_dWc1, n_W1_x.cols[2]) + dot(dL_dWc2, n_W2_x.cols[2]) +
				dot(dL_dWc3, n_W3_x.cols[2]);

	dL_dtau[6 * idx + 3] += dL_dtheta.x;
	dL_dtau[6 * idx + 4] += dL_dtheta.y;
	dL_dtau[6 * idx + 5] += dL_dtheta.z;
}

__device__ void computeCov2DCUDA_backward(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float3* dE_dmeans,
	// const float* dE_dcov3D,
	const float* dE_dtau,
	float4* dE_dconics
	)
{
	// auto idx = cg::this_grid().thread_rank();
	// just the first since we've already shifted the pointer.
	const int idx = 0;
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 t = transformPoint4x3(mean, view_matrix);
	assert(t.z != 0.f);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t

	SE3 T_CW(view_matrix);
	mat33 R = T_CW.R().data();
	mat33 RT = R.transpose();
	float3 t_ = T_CW.t();
	mat33 dpC_drho = mat33::identity();
	mat33 dpC_dtheta = -mat33::skew_symmetric(t);

	float3 c1 = R.cols[0];
	float3 c2 = R.cols[1];
	float3 c3 = R.cols[2];

	mat33 n_W1_x = -mat33::skew_symmetric(c1);
	mat33 n_W2_x = -mat33::skew_symmetric(c2);
	mat33 n_W3_x = -mat33::skew_symmetric(c3);

	// Reflection point
	//!NOTE: mat33 is column-major
	float3 dE_dtheta = {dE_dtau[6 * idx + 3], dE_dtau[6 * idx + 4], dE_dtau[6 * idx + 5]};
	float3 dE_dWc1 = dE_dtheta.x * n_W1_x.cols[0] + dE_dtheta.y * n_W1_x.cols[1] + dE_dtheta.z * n_W1_x.cols[2];
	float3 dE_dWc2 = dE_dtheta.x * n_W2_x.cols[0] + dE_dtheta.y * n_W2_x.cols[1] + dE_dtheta.z * n_W2_x.cols[2];
	float3 dE_dWc3 = dE_dtheta.x * n_W3_x.cols[0] + dE_dtheta.y * n_W3_x.cols[1] + dE_dtheta.z * n_W3_x.cols[2];

	float dE_dW00 = dE_dWc1.x;
	float dE_dW01 = dE_dWc2.x;
	float dE_dW02 = dE_dWc3.x;
	float dE_dW10 = dE_dWc1.y;
	float dE_dW11 = dE_dWc2.y;
	float dE_dW12 = dE_dWc3.y;
	float dE_dW20 = dE_dWc1.z;
	float dE_dW21 = dE_dWc2.z;
	float dE_dW22 = dE_dWc3.z;

	float dE_dT00 = dE_dW00 * J[0][0] + dE_dW20 * J[0][2];
	float dE_dT01 = dE_dW01 * J[0][0] + dE_dW21 * J[0][2];
	float dE_dT02 = dE_dW02 * J[0][0] + dE_dW22 * J[0][2];
	float dE_dT10 = dE_dW10 * J[1][1] + dE_dW20 * J[1][2];
	float dE_dT11 = dE_dW11 * J[1][1] + dE_dW21 * J[1][2];
	float dE_dT12 = dE_dW12 * J[1][1] + dE_dW22 * J[1][2];

	float3 dE_dtxyz = transformVec4x3Transpose_backward({dE_dT00, dE_dT01, dE_dT02}, view_matrix, dE_dmeans[idx]);

	float dE_dtx = dE_dtxyz.x;
	float dE_dty = dE_dtxyz.y;
	float dE_dtz = dE_dtxyz.z;

	// float dE_dt[6];
	for (int i = 0; i < 3; i++) {
		float3 c_rho = dpC_drho.cols[i];
		float3 c_theta = dpC_dtheta.cols[i];

		dE_dtx += dE_dtau[6 * idx + i] * c_rho.x + dE_dtau[6 * idx + i + 3] * c_theta.x;
		dE_dty += dE_dtau[6 * idx + i] * c_rho.y + dE_dtau[6 * idx + i + 3] * c_theta.y;
		dE_dtz += dE_dtau[6 * idx + i] * c_rho.z + dE_dtau[6 * idx + i + 3] * c_theta.z;
	}

	float dE_dJ00 = dE_dtz * (-h_x * tz2);
	float dE_dJ11 = dE_dtz * (-h_y * tz2);
	float dE_dJ02 = dE_dtz * (2 * h_x * t.x) * tz3;
	float dE_dJ12 = dE_dtz * (2 * h_y * t.y) * tz3;
	dE_dJ02 += dE_dtx * x_grad_mul * -h_x * tz2;
	dE_dJ12 += dE_dty * y_grad_mul * -h_y * tz2;

	dE_dT00 += dE_dJ00 * W[0][0] + dE_dJ02 * W[2][0];
	dE_dT01 += dE_dJ00 * W[0][1] + dE_dJ02 * W[2][1];
	dE_dT02 += dE_dJ00 * W[0][2] + dE_dJ02 * W[2][2];

	dE_dT10 += dE_dJ11 * W[1][0] + dE_dJ12 * W[2][0];
	dE_dT11 += dE_dJ11 * W[1][1] + dE_dJ12 * W[2][1];
	dE_dT12 += dE_dJ11 * W[1][2] + dE_dJ12 * W[2][2];

	float dE_da = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dE_dT00 +
		2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dE_dT01 + 
		2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dE_dT02;
	
	float dE_db = 0.f;
	dE_db += (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dE_dT00;
	dE_db += (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dE_dT01;
	dE_db += (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dE_dT02;
	dE_db += (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dE_dT10;
	dE_db += (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dE_dT11;
	dE_db += (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dE_dT12;

	float dE_dc = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dE_dT10 +
		2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dE_dT11 +
		2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dE_dT12;
	
	if (denom2inv != 0){
		dE_dconics[idx].x += denom2inv * (-c * c * dE_da + 2 * b * c * dE_db + (denom - a * c) * dE_dc);
		dE_dconics[idx].y += denom2inv * 2 * (b * c * dE_da - (denom + 2 * b * b) * dE_db + a * b * dE_dc);
		dE_dconics[idx].w += denom2inv * (-a * a * dE_dc + 2 * a * b * dE_db + (denom - a * c) * dE_da);
	}
}

template<int C>
__device__ void preprocessCUDARelocated(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* proj,
	const float* proj_raw,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	const float cur_dL_ddepth,
	const float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dtau)
{
	// auto idx = cg::this_grid().thread_rank();
	int idx = 0;
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	float alpha = 1.0f * m_w;
	float beta = -m_hom.x * m_w * m_w;
	float gamma = -m_hom.y * m_w * m_w;

	float a = proj_raw[0];
	float b = proj_raw[5];
	float c = proj_raw[10];
	float d = proj_raw[14];
	float e = proj_raw[11];

	SE3 T_CW(viewmatrix);
	mat33 R = T_CW.R().data();
	mat33 RT = R.transpose();
	float3 t = T_CW.t();
	float3 p_C = T_CW * m;
	mat33 dp_C_d_rho = mat33::identity();
	mat33 dp_C_d_theta = -mat33::skew_symmetric(p_C);

	float3 d_proj_dp_C1 = make_float3(alpha * a, 0.f, beta * e);
	float3 d_proj_dp_C2 = make_float3(0.f, alpha * b, gamma * e);

	float3 d_proj_dp_C1_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C1; // x.T A = A.T x
	float3 d_proj_dp_C2_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C2;
	float3 d_proj_dp_C1_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C1;
	float3 d_proj_dp_C2_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C2;

	float2 dmean2D_dtau[6];
	dmean2D_dtau[0].x = d_proj_dp_C1_d_rho.x;
	dmean2D_dtau[1].x = d_proj_dp_C1_d_rho.y;
	dmean2D_dtau[2].x = d_proj_dp_C1_d_rho.z;
	dmean2D_dtau[3].x = d_proj_dp_C1_d_theta.x;
	dmean2D_dtau[4].x = d_proj_dp_C1_d_theta.y;
	dmean2D_dtau[5].x = d_proj_dp_C1_d_theta.z;

	dmean2D_dtau[0].y = d_proj_dp_C2_d_rho.x;
	dmean2D_dtau[1].y = d_proj_dp_C2_d_rho.y;
	dmean2D_dtau[2].y = d_proj_dp_C2_d_rho.z;
	dmean2D_dtau[3].y = d_proj_dp_C2_d_theta.x;
	dmean2D_dtau[4].y = d_proj_dp_C2_d_theta.y;
	dmean2D_dtau[5].y = d_proj_dp_C2_d_theta.z;

	float dL_dt[6];
	for (int i = 0; i < 6; i++) {
		dL_dt[i] = dL_dmean2D[idx].x * dmean2D_dtau[i].x + dL_dmean2D[idx].y * dmean2D_dtau[i].y;
	}
	for (int i = 0; i < 6; i++) {
		dL_dtau[6 * idx + i] += dL_dt[i];
	}

	// Compute gradient update due to computing depths
	// p_orig = m
	// p_view = transformPoint4x3(p_orig, viewmatrix);
	// depth = p_view.z;
	// float dL_dpCz = cur_dL_ddepth;
	dL_dmeans[idx].x += cur_dL_ddepth * viewmatrix[2];
	dL_dmeans[idx].y += cur_dL_ddepth * viewmatrix[6];
	dL_dmeans[idx].z += cur_dL_ddepth * viewmatrix[10];

	for (int i = 0; i < 3; i++) {
		float3 c_rho = dp_C_d_rho.cols[i];
		float3 c_theta = dp_C_d_theta.cols[i];
		dL_dtau[6 * idx + i] += cur_dL_ddepth * c_rho.z;
		dL_dtau[6 * idx + i + 3] += cur_dL_ddepth * c_theta.z;
	}

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH_EIG(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh, dL_dtau);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D_EIG(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}


template<int C>
__device__ void preprocessCUDA_backward(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* proj,
	const float* proj_raw,
	const glm::vec3* campos,
	// previous backward
	// const float cur_dL_ddepth,
	const float* dL_dcolor, // Still needed for backward of SH
	// glm::vec3* dL_dscale,
	// glm::vec4* dL_drot,
	// float* dL_dtau,
	// backward input w.r.t dE
	const float3* dE_dmeans,
	const float* dE_dshs,
	const glm::vec3* dE_dscale,
	const glm::vec4* dE_drot,
	const float* dE_dopacity,
	const float* dE_dtaus,
	// backward output
	float* dE_dCov3Ds,
	float* dE_dcolors,
	float* dE_ddepth,
	float3* dE_dmeans2D
	)
{
	// auto idx = cg::this_grid().thread_rank();
	const int idx = 0;
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	const float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	const float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.

	float alpha = 1.0f * m_w;
	float beta = -m_hom.x * m_w * m_w;
	float gamma = -m_hom.y * m_w * m_w;

	float a = proj_raw[0];
	float b = proj_raw[5];
	float c = proj_raw[10];
	float d = proj_raw[14];
	float e = proj_raw[11];

	SE3 T_CW(viewmatrix);
	mat33 R = T_CW.R().data();
	mat33 RT = R.transpose();
	float3 t = T_CW.t();
	float3 p_C = T_CW * m;
	mat33 dp_C_d_rho = mat33::identity();
	mat33 dp_C_d_theta = -mat33::skew_symmetric(p_C);

	float3 d_proj_dp_C1 = make_float3(alpha * a, 0.f, beta * e);
	float3 d_proj_dp_C2 = make_float3(0.f, alpha * b, gamma * e);

	float3 d_proj_dp_C1_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C1; // x.T A = A.T x
	float3 d_proj_dp_C2_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C2;
	float3 d_proj_dp_C1_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C1;
	float3 d_proj_dp_C2_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C2;

	float2 dmean2D_dtau[6];
	dmean2D_dtau[0].x = d_proj_dp_C1_d_rho.x;
	dmean2D_dtau[1].x = d_proj_dp_C1_d_rho.y;
	dmean2D_dtau[2].x = d_proj_dp_C1_d_rho.z;
	dmean2D_dtau[3].x = d_proj_dp_C1_d_theta.x;
	dmean2D_dtau[4].x = d_proj_dp_C1_d_theta.y;
	dmean2D_dtau[5].x = d_proj_dp_C1_d_theta.z;

	dmean2D_dtau[0].y = d_proj_dp_C2_d_rho.x;
	dmean2D_dtau[1].y = d_proj_dp_C2_d_rho.y;
	dmean2D_dtau[2].y = d_proj_dp_C2_d_rho.z;
	dmean2D_dtau[3].y = d_proj_dp_C2_d_theta.x;
	dmean2D_dtau[4].y = d_proj_dp_C2_d_theta.y;
	dmean2D_dtau[5].y = d_proj_dp_C2_d_theta.z;

	float cur_dL_dcov3D[6];
	// Compute gradient updates due to computing covariance from scale/rotation
	glm::vec3 cur_dL_dscale(0.0f, 0.0f, 0.0f);
	glm::vec4 cur_dL_drot(0.0f, 0.0f, 0.0f, 0.0f);
	if (scales)
		computeCov3D_EIG(idx, scales[idx], scale_modifier, rotations[idx], cur_dL_dcov3D, &cur_dL_dscale, &cur_dL_drot);
	
	// Reflection point:

	// no need to compute 
	// float dE_dCov3Ds[6];
	if (scales)
		computeCov3D_EIG_backward(idx, scales[idx], scale_modifier, rotations[idx], (const float*) cur_dL_dcov3D, 
			dE_drot[idx], dE_dscale[idx], dE_dCov3Ds);

	if (shs)
		computeColorFromSH_backward(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (const glm::vec3*)dL_dcolor, (const glm::vec3*)dE_dmeans, dE_dshs, dE_dtaus, dE_dcolors);

	for (int i = 0; i < 3; i++) {
		float3 c_rho = dp_C_d_rho.cols[i];
		float3 c_theta = dp_C_d_theta.cols[i];
		//!NOTE: this code could only work with idx=0
		dE_ddepth[idx] += dE_dtaus[6 * idx + i] * c_rho.z + dE_dtaus[6 * idx + i + 3] * c_theta.z;
	}

	dE_ddepth[idx] += dE_dmeans[idx].x * viewmatrix[2] + dE_dmeans[idx].y * viewmatrix[6] + dE_dmeans[idx].z * viewmatrix[10];

	for (int i = 0; i < 6; i++) {
		dE_dmeans2D->x += dE_dtaus[6 * idx + i] * dmean2D_dtau[i].x;
		dE_dmeans2D->y += dE_dtaus[6 * idx + i] * dmean2D_dtau[i].y;
	}

	// float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	// float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;

	dE_dmeans2D[idx].x = dE_dmeans[idx].x * (proj[0] * m_w - proj[3] * mul1) + dE_dmeans[idx].y * (proj[4] * m_w - proj[7] * mul1) + dE_dmeans[idx].z * (proj[8] * m_w - proj[11] * mul1);
	dE_dmeans2D[idx].y = dE_dmeans[idx].x * (proj[1] * m_w - proj[3] * mul2) + dE_dmeans[idx].y * (proj[5] * m_w - proj[7] * mul2) + dE_dmeans[idx].z * (proj[9] * m_w - proj[11] * mul2);

}

template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
fisher_backward_kernel(
	// params for render function
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	int P, int D, const int M,
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
	const float* dE_dshs,
	const glm::vec3* dE_dscale,
	const glm::vec4* dE_drot,
	const float* dE_dtaus,
	// output
	float* dE_dpixels,
	float* dE_dpixels_depth,
    const int grad_power)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	// miss tid here

	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x; // 895 - 908 checked

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE]; // depth buffer

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C] = { 0 };
	float accum_rec_depth = 0;
	float dL_dpixel_depth = 0;

	float dE_dpixel[C] = { 0.f };
	float dE_dpixel_depth = 0.f;
	glm::vec3 cur_dE_dshs[MAX_SH_COEFFS];
	assert(M <= MAX_SH_COEFFS);

	if (inside){
		#pragma unroll
		for (int i = 0; i < C; i++) {
			// dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
			// Simulate the backward pass
			dL_dpixel[i] = 1.f;
		}
		dL_dpixel_depth = 1.f; 
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0.f; // accumulated depth

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;
	// __shared__ int skip_counter;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank(); // replace tid here
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// NOTE: We need to go the first forward round first because we can't afford
			// saving intermediate results in GPU memory
			// TODO: reverse this process to compute dE_dx

			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// prepare intermediate values for backwards
			// this sounds we will initalize many times, but let optimizer move this out of the loop
			glm::vec3 cur_dL_dcolors(0.0f, 0.0f, 0.0f);
			glm::vec3 cur_dL_dmean2D(0.0f, 0.0f, 0.0f);
			float4 cur_dL_dconic2D = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // only x y w are used
			float cur_dL_dopacity = 0.0f;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				cur_dL_dcolors[ch] = dchannel_dcolor * dL_dchannel;
				// atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			// compute gradient w.r.t depth
			const float depth = collected_depths[j];
			accum_rec_depth = last_alpha * last_depth + (1.f - last_alpha) * accum_rec_depth;
			last_depth = depth;
			dL_dalpha += (depth - accum_rec_depth) * dL_dpixel_depth; 	// additional gradient from depth
			float cur_dL_ddepth = dchannel_dcolor * dL_dpixel_depth;		// depth is just another channel from color

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// TODO: Add dL_dmean2D and dL_dconic2D as local variables and 
			// invoke a function call to calcultate graidents for all the variales
			// in this thread

			// Update gradients w.r.t. 2D mean position of the Gaussian
			cur_dL_dmean2D.x = dL_dG * dG_ddelx * ddelx_dx;
			cur_dL_dmean2D.y = dL_dG * dG_ddely * ddely_dy;

			cur_dL_dconic2D.x = -0.5f * gdx * d.x * dL_dG;
			cur_dL_dconic2D.y = -0.5f * gdx * d.y * dL_dG;
			cur_dL_dconic2D.w = -0.5f * gdy * d.y * dL_dG;
			cur_dL_dopacity = G * dL_dalpha;

			// Start to backward to other variables
			if (global_id >= P)
				continue;

			glm::vec3 cur_dL_dmeans(0.f);
			float cur_dL_dcov3D[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,};
			float cur_dL_dtau[6] = {0.f};
			computeCov2DCUDARelocated_EIG(P,
				&means3D[global_id],
				&radii[global_id],
				cov3Ds + 6 * global_id,
				focal_x,
				focal_y,
				tan_fovx,
				tan_fovy,
				viewmatrix,
				(const float*)&cur_dL_dconic2D,
				(float3* )&cur_dL_dmeans,
				(float*)cur_dL_dcov3D,
				cur_dL_dtau);

			// Spheical harmonics degree is (1+deg)**2
			const int num_shs = (1 + D) * (1 + D);
			const int MAX_SH_CHANNELS = 16;
			float3 cur_dL_dshs[MAX_SH_CHANNELS];
			memset(cur_dL_dshs, 0,  sizeof(float3) * MAX_SH_CHANNELS);
			// float 0 should be equal to 0.0f
			const glm::vec3* cur_scales = scales ? &scales[global_id] : scales;
			const float* cur_shs = shs ? shs + M * global_id : shs;

			glm::vec3 cur_dL_dscale(0.0f, 0.0f, 0.0f);
			glm::vec4 cur_dL_drot(0.0f, 0.0f, 0.0f, 0.0f);

   			// float cur_dL_dtau[6] = {0.f};
			// cur_dL_dcov
			preprocessCUDARelocated<NUM_CHANNELS>(
				P, D, M,
				&means3D[global_id],
				&radii[global_id],
				cur_shs,
				&clamped[3 * global_id],	
				cur_scales,
				&rotations[global_id],
				scale_modifier,
				viewmatrix,
				projmatrix,
				projmatrix_raw,
				campos,
				(const float3*)&cur_dL_dmean2D,
				(glm::vec3*)&cur_dL_dmeans,
				cur_dL_ddepth,
				(const float*)&cur_dL_dcolors,
				(float*)cur_dL_dcov3D,
				(float*)cur_dL_dshs,
				&cur_dL_dscale,
				&cur_dL_drot,
				cur_dL_dtau
			);
#if 1

			// start the real backward pass
			//!NOTE: Reflection point
			glm::vec3 cur_dE_dmeans = 2.f * cur_dL_dmeans * dE_dmean3D[global_id];
			glm::vec3* cur_vec_dE_dshs = (glm::vec3*) dE_dshs + global_id * num_shs;
            if(D > 0){
                for(int ch = 0; ch < num_shs; ch++)
                {
					cur_dE_dshs[ch] = 2.f * ((glm::vec3*) cur_dL_dshs)[ch] * cur_vec_dE_dshs[ch];
                }
            }

			glm::vec3 cur_dE_dscale = 2.f * cur_dL_dscale * dE_dscale[global_id];
			glm::vec4 cur_dE_drot = 2.f * cur_dL_drot * dE_drot[global_id];
			float cur_dE_dopacity = 2.f * cur_dL_dopacity * dE_dopacity[global_id];

			float cur_dE_dCov3Ds[6];
			float cur_dE_dcolors[C];
			float cur_dE_depth = 0.0f;
			float3 cur_dE_dmeans2D = make_float3(0.0f, 0.0f, 0.0f);

			preprocessCUDA_backward<NUM_CHANNELS>(
				P, D, M,
				&means3D[global_id],
				&radii[global_id],
				cur_shs,
				&clamped[3 * global_id],	
				cur_scales,
				&rotations[global_id],
				scale_modifier,
				viewmatrix,
				projmatrix,
				projmatrix_raw,
				campos,
				// (const float3*)&cur_dL_dmean2D,
				(const float*) &cur_dL_dcolors,
				(const float3*) &cur_dE_dmeans,
				(const float*) cur_dE_dshs,
				&cur_dE_dscale,
				&cur_dE_drot,
				&cur_dE_dopacity,
				(const float*) dE_dtaus + 6 * global_id,
				// outputs
				cur_dE_dCov3Ds,
				cur_dE_dcolors,
				&cur_dE_depth,
				&cur_dE_dmeans2D
			);

			float4 cur_dE_dconic2D;
			computeCov2DCUDA_backward(P,
				&means3D[global_id],
				&radii[global_id],
				cov3Ds + 6 * global_id,
				focal_x,
				focal_y,
				tan_fovx,
				tan_fovy,
				viewmatrix,
				(const float3 *) &cur_dE_dmeans,
				// cur_dE_dCov3Ds,
				(const float*) dE_dtaus + 6 * global_id,
				&cur_dE_dconic2D);
			
			float cur_dE_dalpha = cur_dE_dopacity * G;

			float cur_dE_dG = 0.f;
			cur_dE_dG += -0.5f * gdx * d.x * cur_dE_dconic2D.x;
			cur_dE_dG += -0.5f * gdx * d.y * cur_dE_dconic2D.y;
			cur_dE_dG += -0.5f * gdy * d.y * cur_dE_dconic2D.w;

			cur_dE_dG += cur_dE_dmeans2D.x * dG_ddelx * ddelx_dx;
			cur_dE_dG += cur_dE_dmeans2D.y * dG_ddely * ddely_dy;

			cur_dE_dalpha += con_o.w * cur_dE_dG;

			// float3 dE_dpixel = make_float3(0.0f, 0.0f, 0.0f);
			float dE_dbg_dot_dpixel = (-T_final / (1.f - alpha)) * cur_dE_dalpha;
			for (int i = 0; i < C; i++)
				dE_dpixel[i] += bg_color[i] * dE_dbg_dot_dpixel;

			cur_dE_dalpha *= T;

			dE_dpixel_depth += (depth - accum_rec_depth) * cur_dE_dalpha;

			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				dE_dpixel[ch] += (c - accum_rec[ch]) * cur_dE_dalpha;				
			}
#endif
		}
	}
	if (inside){
		#pragma unroll
		for (int i = 0; i < C; i++) {
			dE_dpixels[i * H * W + pix_id] = dE_dpixel[i];
		}
		dE_dpixels_depth[pix_id] = dE_dpixel_depth;
	}

}

// Fused kernel for the whole backward process
void FISHER_BACKWARD::backward(
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
		const float* dE_dshs,
		const glm::vec3* dE_dscale,
		const glm::vec4* dE_drot,
		const float* dE_dtaus,
		// output
		float* dE_dpixels,
		float* dE_dpixels_depth,
		const int grad_power
	)
{
	fisher_backward_kernel<NUM_CHANNELS> <<<grid, block >>> (
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depth,
		final_Ts,
		n_contrib,
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3Ds,
		viewmatrix,
		projmatrix,
		projmatrix_raw,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		campos,
		dE_dmean3D,
		dE_dopacity,
		dE_dshs,
		dE_dscale,
		dE_drot,
		dE_dtaus,
		dE_dpixels,
		dE_dpixels_depth,
        grad_power);
}
