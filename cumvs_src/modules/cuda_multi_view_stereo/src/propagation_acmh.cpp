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

#include "propagation_acmh.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>

#include "debug_print.h"

#include "cuda_multi_view_stereo_internal.h"

namespace cv
{
namespace cuda
{

class PropagationACMHImpl : public PropagationACMH
{
public:

	PropagationACMHImpl(const std::vector<Mat>& Is, const std::vector<Matx33d>& Ks, const std::vector<Matx33d>& Rs,
		const std::vector<Vec3d>& ts, const std::vector<Mat>& Ds, float minZ, float maxZ, bool geom)
		: minZ_(minZ), maxZ_(maxZ), geom_(geom)
	{
		nviews_ = static_cast<int>(Is.size());
	}

	void calcInitialCosts(const GpuMat& depths, const GpuMat& normals, GpuMat& costs) override
	{
		CV_Assert(depths.type() == CV_32F);
		CV_Assert(normals.type() == CV_32FC3);

		gpu::calcInitialCosts(depths, normals, costs, nviews_, geom_);
	}

	void propagateAndRefine(GpuMat& depths, GpuMat& normals, GpuMat& costs, int color) override
	{
		TickMeter t;
		t.start();

		gpu::propagateAndRefineACMH(depths, normals, costs, nviews_, color, minZ_, maxZ_, geom_);
		gpu::waitForKernelCompletion();

		t.stop();
		// DEBUG_PRINT("propagateAndRefine: %7.1f[msec]\n", t.getTimeMilli());
	}

	void next() override {}

private:

	float minZ_;
	float maxZ_;
	int nviews_;
	bool geom_;
};

Ptr<PropagationACMH> PropagationACMH::create(const std::vector<Mat>& Is, const std::vector<Matx33d>& Ks, const std::vector<Matx33d>& Rs,
	const std::vector<Vec3d>& ts, const std::vector<Mat>& Ds, float minZ, float maxZ, bool geom)
{
	return makePtr<PropagationACMHImpl>(Is, Ks, Rs, ts, Ds, minZ, maxZ, geom);
}

PropagationACMH::~PropagationACMH()
{
}

} // namespace cuda
} // namespace cv
