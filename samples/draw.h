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

#ifndef __DRAW_H__
#define __DRAW_H__

#include <opencv2/core.hpp>
#include <opencv2/viz.hpp>

namespace cv
{

Mat colored(const Mat& src, const Mat& zeroMask = Mat());

Mat coloredNormal(const Mat& normals);

void drawViewConnections(viz::Viz3d& window, InputArrayOfArrays cameras, InputArrayOfArrays rotations,
	InputArrayOfArrays translations, InputArrayOfArrays viewIdSets);

void drawReconstruction(viz::Viz3d& window, InputArrayOfArrays cameras, InputArrayOfArrays rotations,
	InputArrayOfArrays translations, InputArray points3D, InputArray colors3D, InputArray normals3D = noArray(), int normalLevel = 64);

} // namespace cv

#endif // !__DRAW_H__
