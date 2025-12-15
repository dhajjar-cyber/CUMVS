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

#ifndef __PROPAGATION_ACMH_H__
#define __PROPAGATION_ACMH_H__

#include "propagation.h"

namespace cv
{
namespace cuda
{

class PropagationACMH : public RedBlackPropagation
{
public:

	static Ptr<PropagationACMH> create(const std::vector<Mat>& Is, const std::vector<Matx33d>& Ks, const std::vector<Matx33d>& Rs,
		const std::vector<Vec3d>& ts, const std::vector<Mat>& Ds, float minZ, float maxZ, bool geom);
	virtual ~PropagationACMH();
};

} // namespace cuda
} // namespace cv

#endif // !__PROPAGATION_ACMH_H__
