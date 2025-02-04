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

#ifndef __PROPAGATION_H__
#define __PROPAGATION_H__

#include <opencv2/core.hpp>

namespace cv
{
namespace cuda
{

enum
{
	PIXEL_COLOR_BLK = 0,
	PIXEL_COLOR_RED = 1
};

class RedBlackPropagation
{
public:

	virtual void calcInitialCosts(const GpuMat& depths, const GpuMat& normals, GpuMat& costs) = 0;
	virtual void propagateAndRefine(GpuMat& depths, GpuMat& normals, GpuMat& costs, int color) = 0;
	virtual void next() = 0;
	virtual ~RedBlackPropagation();
};

} // namespace cuda
} // namespace cv

#endif // !__PROPAGATION_H__
