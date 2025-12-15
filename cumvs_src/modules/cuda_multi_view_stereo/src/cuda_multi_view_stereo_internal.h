/*
Copyright 2025 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef __CUDA_MULTI_VIEW_STEREO_INTERNAL_H__
#define __CUDA_MULTI_VIEW_STEREO_INTERNAL_H__

#include <opencv2/core.hpp>
#include <texture_types.h>

namespace cv
{
namespace cuda
{
namespace gpu
{

struct TexObjSz
{
	cudaTextureObject_t obj;
	int w, h;
};

void waitForKernelCompletion();
void uploadImages(const std::vector<TexObjSz>& images);
void uploadDepths(const std::vector<TexObjSz>& depths);
void uploadHomographyParams(const Vec4f& invK1, const std::vector<Matx33f>& R21, const std::vector<Vec3f>& t21);
void uploadHomographyParams(const Vec4f& invK1, const std::vector<Matx33f>& R21, const std::vector<Vec3f>& t21,
	const std::vector<Matx33f>& R12, const std::vector<Vec3f>& t12);
void propagateAndRefineGipuma(GpuMat& depths, GpuMat& normals, GpuMat& costs, int nviews, int color,
	float minZ, float maxZ, float deltaDepth, float deltaAngle, int niterations, bool geom);
void propagateAndRefineACMH(GpuMat& depths, GpuMat& normals, GpuMat& costs, int nviews, int color,
	float minZ, float maxZ, bool geom);
void calcInitialCosts(const GpuMat& depths, const GpuMat& normals, GpuMat& costs, int nviews, bool geom);
void initializeDepthsAndNormals(GpuMat& depths, GpuMat& normals, float minZ, float maxZ);

} // namespace gpu
} // namespace cuda
} // namespace cv

#endif // !__CUDA_MULTI_VIEW_STEREO_INTERNAL_H__
