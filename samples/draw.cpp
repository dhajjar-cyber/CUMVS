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

#include "draw.h"

#include <opencv2/imgproc.hpp>

namespace cv
{

Mat colored(const Mat& src, const Mat& zeroMask)
{
	Mat tmp, dst;
	normalize(src, tmp, 0, 255, NORM_MINMAX, CV_8U);
	applyColorMap(tmp, dst, COLORMAP_TURBO);
	if (!zeroMask.empty())
		dst.setTo(0, zeroMask);
	return dst;
}

Mat coloredNormal(const Mat& normals)
{
	Mat dst;
	normals.convertTo(dst, CV_8UC3, -255.f / 2.f, 255.f / 2.f);
	return dst;
}

void drawViewConnections(viz::Viz3d& window, InputArrayOfArrays _cameras, InputArrayOfArrays _rotations,
	InputArrayOfArrays _translations, InputArrayOfArrays _viewIdSets)
{

	std::vector<Mat> cameras, rotations, translations, viewIdSets;
	_cameras.getMatVector(cameras);
	_rotations.getMatVector(rotations);
	_translations.getMatVector(translations);
	_viewIdSets.getMatVector(viewIdSets);

	RNG random(1);
	for (const auto& viewIds : viewIdSets)
	{
		const auto color = viz::Color(random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255));
		const int nviews = viewIds.checkVector(1);
		const int i0 = viewIds.at<int>(0);

		Matx33d K0 = cameras[i0].reshape(1, 3);
		Matx33d R0 = rotations[i0].reshape(1, 3);
		Vec3d t0 = translations[i0].reshape(1, 3);

		const std::string widgetName = format("View %3d", i0);
		viz::WCameraPosition camera(Matx33d(K0), 0.5, color);
		window.showWidget(widgetName, camera);
		window.setWidgetPose(widgetName, Affine3d(R0, t0));

		for (int i = 0; i < nviews; i++)
		{
			const int i1 = viewIds.at<int>(i);
			Vec3d t1 = translations[i1].reshape(1, 3);
			Vec3d tm = 0.5 * (t0 + t1);
			viz::WLine line(t0, t1, viz::Color::white());
			viz::WArrow arrow(t0, tm, 1e-2 / norm(tm - t0), color);
			window.showWidget(format("View %3d Neighbor %3d 0", i0, i1), line);
			window.showWidget(format("View %3d Neighbor %3d 1", i0, i1), arrow);
		}
	}
}

void drawReconstruction(viz::Viz3d& window, InputArrayOfArrays _cameras, InputArrayOfArrays _rotations,
	InputArrayOfArrays _translations, InputArray points3D, InputArray colors3D, InputArray normals3D, int normalLevel)
{
	if (!normals3D.empty() && normalLevel > 0)
	{
		viz::WCloudNormals normals(points3D, normals3D, normalLevel, 0.1, viz::Color::red());
		window.showWidget("normals", normals);
	}

	viz::WCloud cloud(points3D, colors3D);
	window.showWidget("cloud", cloud);

	std::vector<Mat> cameras, rotations, translations;
	_cameras.getMatVector(cameras);
	_rotations.getMatVector(rotations);
	_translations.getMatVector(translations);

	const int nimages = static_cast<int>(cameras.size());
	for (int i = 0; i < nimages; i++)
	{
		Mat K = cameras[i].reshape(1, 3);
		Mat R = rotations[i].reshape(1, 3);
		Mat t = translations[i].reshape(1, 3);

		const std::string widgetName = format("View %3d", i);
		viz::WCameraPosition camera(Matx33d(K), 0.5);
		window.showWidget(widgetName, camera);
		window.setWidgetPose(widgetName, Affine3d(R, t));
	}
}

} // namespace cv
