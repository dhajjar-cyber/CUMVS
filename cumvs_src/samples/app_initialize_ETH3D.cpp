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

#include <iostream>
#include <fstream>
#include <algorithm>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>

#include "io.h"
#include "draw.h"

static std::string keys =
"{ @input-directory | <none> | input directory.                             }"
"{ output-directory | ./work | output directory.                            }"
"{ min-neighbors    |      2 | minimum number of neighbor images.           }"
"{ max-neighbors    |     20 | maximum number of neighbor images.           }"
"{ num-threads      |      8 | number of threads for parallel computations. }"
"{ debug d          |        | enable debug view.                           }"
"{ help  h          |        | print help message.                          }";

namespace cv
{

static std::vector<std::string> getLines(const std::string& filename)
{
	std::ifstream ifs(filename);
	CV_Assert(!ifs.fail());
	std::vector<std::string> lines;
	std::string str;
	while (std::getline(ifs, str))
		if (str[0] != '#')
			lines.push_back(str);
	return lines;
}

static std::vector<std::string> split(const std::string& line)
{
	std::istringstream iss(line);
	std::string str;
	std::vector<std::string> strs;
	while (std::getline(iss, str, ' '))
		strs.push_back(str);
	return strs;
}

static void loadETH3D(const std::string& inputDir, std::vector<std::string>& images, std::vector<Matx33d>& cameras,
	std::vector<Matx33d>& rotations, std::vector<Vec3d>& translations, std::vector<Vec3d>& objectPoints,
	std::vector<Vec3b>& objectColors, std::vector<std::vector<int>>& imageToPointIds)
{
	images.clear();
	cameras.clear();
	rotations.clear();
	translations.clear();
	objectPoints.clear();
	objectColors.clear();
	imageToPointIds.clear();

	// load camera parameters
	const auto cameraLines = getLines(inputDir + "/dslr_calibration_undistorted/cameras.txt");
	std::map<int, Matx33d> cameraTable;
	for (const auto& line : cameraLines)
	{
		int cameraId, width, height;
		double fx, fy, cx, cy;
		char model[256];

		std::sscanf(line.c_str(), "%d %s %d %d %lf %lf %lf %lf",
			&cameraId, model, &width, &height, &fx, &fy, &cx, &cy);

		cameraTable[cameraId] = Matx33d(fx, 0, cx, 0, fy, cy, 0, 0, 1);
	}

	// load points
	const auto pointLines = getLines(inputDir + "/dslr_calibration_undistorted/points3D.txt");
	int npoints = 0;
	std::map<int, int> pointIdTable;
	for (const auto& line : pointLines)
	{
		const auto strs = split(line);

		const auto id = std::stoi(strs[0]);

		const auto X = std::stod(strs[1]);
		const auto Y = std::stod(strs[2]);
		const auto Z = std::stod(strs[3]);

		const auto R = std::stoi(strs[4]);
		const auto G = std::stoi(strs[5]);
		const auto B = std::stoi(strs[6]);

		objectPoints.push_back(Vec3d(X, Y, Z));
		objectColors.push_back(Vec3b(B, G, R));
		pointIdTable[id] = npoints++;
	}

	// load images and poses
	const auto imageLines = getLines(inputDir + "/dslr_calibration_undistorted/images.txt");
	for (size_t i = 0; i < imageLines.size(); i += 2)
	{
		// parse first line
		int imageId, cameraId;
		double qw, qx, qy, qz, tx, ty, tz;
		char name[256];

		std::sscanf(imageLines[i].c_str(), "%d %lf %lf %lf %lf %lf %lf %lf %d %s",
			&imageId, &qw, &qx, &qy, &qz, &tx, &ty, &tz, &cameraId, name);

		const Matx33d R = Quatd(qw, qx, qy, qz).toRotMat3x3();
		const Vec3d t(tx, ty, tz);

		images.push_back(inputDir + "/images/" + name);
		cameras.push_back(cameraTable.at(cameraId));
		rotations.push_back(R.t());
		translations.push_back(-R.t() * t);

		// parse second line
		const auto strs = split(imageLines[i + 1]);
		std::vector<int> pointIds;
		pointIds.reserve(strs.size() / 3);
		for (size_t i = 0; i < strs.size(); i += 3)
		{
			const int pointId = std::stoi(strs[i + 2]);
			if (pointId >= 0)
				pointIds.push_back(pointIdTable[pointId]);
		}
		std::sort(std::begin(pointIds), std::end(pointIds));
		imageToPointIds.push_back(pointIds);
	}
}

static void selectNeighborViews(const std::vector<Matx33d>& rotations, const std::vector<Vec3d>& translations,
	const std::vector<Vec3d>& objectPoints, const std::vector<std::vector<int>>& imageToPointIds,
	std::vector<Mat>& viewIdSets, int minNeighbors = 2, int maxNeighbors = 20)
{
	viewIdSets.clear();

	const int nimages = static_cast<int>(rotations.size());

	Mat1s scores(nimages, nimages);
	scores = 0;

	for (int imageId1 = 0; imageId1 < nimages - 1; imageId1++)
	{
		for (int imageId2 = imageId1 + 1; imageId2 < nimages; imageId2++)
		{
			const auto& pointIds1 = imageToPointIds[imageId1];
			const auto& pointIds2 = imageToPointIds[imageId2];
			std::vector<int> intersection;
			std::set_intersection(std::begin(pointIds1), std::end(pointIds1), std::begin(pointIds2), std::end(pointIds2),
				std::back_inserter(intersection));

			if (intersection.empty())
				continue;

			const auto& O1 = translations[imageId1];
			const auto& O2 = translations[imageId2];
			const int npoints = static_cast<int>(intersection.size());
			std::vector<double> angles(npoints);
			for (int i = 0; i < npoints; i++)
			{
				const auto& Xw = objectPoints[intersection[i]];
				const auto v1 = O1 - Xw;
				const auto v2 = O2 - Xw;
				angles[i] = (180 / CV_PI) * ::acos(v1.dot(v2) / (norm(v1) * norm(v2)));
			}

			std::sort(std::begin(angles), std::end(angles));
			const double testAngle = angles[3 * npoints / 4];
			if (testAngle < 1)
				continue;

			scores(imageId1, imageId2) = npoints;
			scores(imageId2, imageId1) = npoints;
		}
	}

	for (int referenceId = 0; referenceId < nimages; referenceId++)
	{
		std::vector<std::pair<int, int>> neighborIds;
		for (int neighborId = 0; neighborId < nimages; neighborId++)
		{
			const int score = scores(referenceId, neighborId);
			if (scores(referenceId, neighborId) > 0)
				neighborIds.push_back({ score, neighborId });
		}

		if (static_cast<int>(neighborIds.size()) > maxNeighbors)
		{
			std::sort(std::begin(neighborIds), std::end(neighborIds), std::greater<std::pair<int, int>>());
			neighborIds.resize(maxNeighbors);
		}

		const int nneighbors = static_cast<int>(neighborIds.size());
		const int nviews = 1 + nneighbors;

		if (nneighbors < minNeighbors)
			continue;

		Mat viewIds(1, nviews, CV_32S);
		viewIds.at<int>(0) = referenceId;
		for (int i = 0; i < nneighbors; i++)
			viewIds.at<int>(i + 1) = neighborIds[i].second;
		viewIdSets.push_back(viewIds);

		std::cout << "view indices [r, n1, n2, ...]   : " << viewIds << std::endl;
	}
}

} // namespace cv

int main(int argc, char* argv[])
{
	using namespace cv;

	const CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	// get parameters
	const auto inputDir = parser.get<std::string>("@input-directory");
	const auto outputDir = parser.get<std::string>("output-directory");
	const auto minNeighbors = parser.get<int>("min-neighbors");
	const auto maxNeighbors = parser.get<int>("max-neighbors");

	if (!parser.check())
	{
		parser.printErrors();
		parser.printMessage();
		std::exit(EXIT_FAILURE);
	}

	std::cout << "===== configulations =====" << std::endl;
	std::cout << "input directory       : " << inputDir << std::endl;
	std::cout << "min neighbor images   : " << minNeighbors << std::endl;
	std::cout << "max neighbor images   : " << maxNeighbors << std::endl;
	std::cout << "==========================" << std::endl << std::endl;

	// load input data
	std::vector<std::string> images;
	std::vector<Matx33d> cameras;
	std::vector<Matx33d> rotations;
	std::vector<Vec3d> translations;
	std::vector<Vec3d> sparsePoints;
	std::vector<Vec3b> sparseColors;
	std::vector<std::vector<int>> imageToPointIds;

	loadETH3D(inputDir, images, cameras, rotations, translations, sparsePoints, sparseColors, imageToPointIds);

	std::cout << "===== data profile =====" << std::endl;
	std::cout << "number of images         : " << images.size() << std::endl;
	std::cout << "number of initial points : " << sparsePoints.size() << std::endl;

	// for each reference view, select neighbor images
	std::vector<Mat> viewIdSets;
	selectNeighborViews(rotations, translations, sparsePoints, imageToPointIds,
		viewIdSets, minNeighbors, maxNeighbors);

	// save data
	makeDirectory(outputDir);

	saveImageInfo(outputDir + "/input_images.json", images, cameras);
	saveGlobalPoses(outputDir + "/global_poses.json", rotations, translations);
	saveViewIdSets(outputDir + "/view_id_sets.json", viewIdSets);
	viz::writeCloud(outputDir + "/point_cloud_sparse.ply", sparsePoints, sparseColors);

	// debug view
	if (parser.has("debug"))
	{
		viz::Viz3d window("view connections");
		drawViewConnections(window, cameras, rotations, translations, viewIdSets);
		window.spin();
	}

	return 0;
}
