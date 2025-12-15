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
#include <map>
#include <numeric>
#include <algorithm>
#include <random>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>

#include <omp.h>

#include <cuda_multi_view_stereo.h>

#include "io.h"
#include "draw.h"

static std::string keys =
"{ @input-directory | <none> | input directory.                                                            }"
"{ output-directory | output | output directory.                                                           }"
"{ method           |      1 | PatchMatch method (0:GIPUMA_FAST 1:ACMH).                                   }"
"{ pm-iterations    |      3 | number of innier PatchMatch iterations.                                     }"
"{ max-image-size   |   3200 | maximum width or height of processed images.                                }"
"{ gc-iterations    |      2 | number of outer PatchMatch iterations performed with geometric consistency. }"
"{ multi-scale      |      1 | whether to use multi scale.                                                 }"
"{ max-points       |      0 | maximum number of points in output ply (0: unlimited).                      }"
"{ normal-level     |      0 | display only every level th normal (0:disable draw).                        }"
"{ num-threads      |     -1 | number of threads for CPU processing (-1:auto detect).                      }"
"{ debug d          |        | enable debug view.                                                          }"
"{ help  h          |        | print help message.                                                         }";

namespace cv
{

static void debugCallBack(InputArray _image, InputArray _depths, InputArray _normals, InputArray _costs, const String& title)
{
	auto getMat = [](InputArray arr) -> Mat
	{
		Mat mat;
		if (arr.kind() == _InputArray::MAT)
			mat = arr.getMat();
		else if (arr.kind() == _InputArray::CUDA_GPU_MAT)
			arr.getGpuMat().download(mat);
		else
			CV_Error(Error::StsBadArg, "Unsupported InputArray kind");
		return mat;
	};

	const Mat image = getMat(_image);
	const Mat depths = getMat(_depths);
	const Mat normals = getMat(_normals);
	const Mat costs = getMat(_costs);

	const Mat depthColored = colored(depths);
	const Mat normalColored = coloredNormal(normals);
	const Mat costsColored = colored(costs);

	rectangle(depthColored, Rect(10, 10, 500, 70), Scalar::all(255), -1);
	putText(depthColored, format("%s", title.c_str()), Point(20, 40), 1, 2, Scalar(0, 0, 255), 2);
	putText(depthColored, format("Total cost: %.1f", sum(costs)[0]), Point(20, 70), 1, 2, Scalar(0, 0, 255), 2);

	imshow("input image", image);
	imshow("depth map", depthColored);
	imshow("normal map", normalColored);
	imshow("cost map", costsColored);
	waitKey(1);
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
	const auto method = parser.get<int>("method");
	const auto pmIters = parser.get<int>("pm-iterations");
	const auto maxImageSize = parser.get<int>("max-image-size");
	const auto gcIters = parser.get<int>("gc-iterations");
	const auto multiScale = parser.get<int>("multi-scale") == 1;
	const auto maxPoints = parser.get<int>("max-points");
	const auto debugView = parser.has("debug");
	const auto normalLevel = parser.get<int>("normal-level");
	auto nthreads = parser.get<int>("num-threads");

	if (!parser.check())
	{
		parser.printErrors();
		parser.printMessage();
		std::exit(EXIT_FAILURE);
	}

	if (nthreads < 0)
		nthreads = getNumThreads();

	const char* methodStr[] = { "GIPUMA_FAST", "ACMH" };

	std::cout << "==================== configulations ====================" << std::endl;
	std::cout << "input directory                  : " << inputDir << std::endl;
	std::cout << "output directory                 : " << outputDir << std::endl;
	std::cout << "PatchMatch method                : " << methodStr[method] << std::endl;
	std::cout << "PatchMatch iterations            : " << pmIters << std::endl;
	std::cout << "maximum image size               : " << maxImageSize << std::endl;
	std::cout << "geometric consistency iterations : " << gcIters << std::endl;
	std::cout << "multi scale                      : " << (multiScale ? "Yes" : "No") << std::endl;
	std::cout << "debug view                       : " << (debugView ? "Yes" : "No") << std::endl;
	std::cout << "normal level                     : " << normalLevel << std::endl;
	std::cout << "number of threads                : " << nthreads << std::endl;
	std::cout << "========================================================" << std::endl << std::endl;

	// set number of threads
	omp_set_num_threads(nthreads);

	// load input data
	std::vector<Mat> images, cameras, rotations, translations, viewIdSets;
	Mat sparsePoints, sparseColors;

	// load input data
	loadImagesAndCameras(inputDir + "/input_images.json", images, cameras, maxImageSize);
	loadGlobalPoses(inputDir + "/global_poses.json", rotations, translations);
	loadViewIdSets(inputDir + "/view_id_sets.json", viewIdSets);
	sparsePoints = viz::readCloud(inputDir + "/point_cloud_sparse.ply", sparseColors);

	// run PatchMatch
	auto mvs = cuda::PatchMatchMVS::create();
	mvs->setPatchMatchMethod(static_cast<cv::cuda::PatchMatchMethod>(method));
	mvs->setPatchMatchIters(pmIters);
	mvs->setGeometricConsistencyIters(gcIters);
	mvs->setMultiScale(multiScale);
	if (debugView)
		mvs->setDebugCallBack(debugCallBack);

	Mat densePoints, denseColors, denseNormals;
	mvs->compute(images, cameras, rotations, translations, viewIdSets, sparsePoints,
		densePoints, denseColors, denseNormals);

	std::cout << "number of initial points : " << sparsePoints.size() << std::endl;
	std::cout << "number of dense points   : " << densePoints.size() << std::endl;

	if (maxPoints > 0 && densePoints.total() > (size_t)maxPoints)
	{
		// Ensure N x 1 format for row-based access
		if (densePoints.rows == 1 && densePoints.cols > 1) {
			densePoints = densePoints.reshape(3, densePoints.cols);
			denseColors = denseColors.reshape(3, denseColors.cols);
			if (!denseNormals.empty())
				denseNormals = denseNormals.reshape(3, denseNormals.cols);
		}

		std::cout << "Subsampling point cloud to " << maxPoints << " points..." << std::endl;
		
		std::vector<int> indices(densePoints.total());
		std::iota(indices.begin(), indices.end(), 0);
		
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(indices.begin(), indices.end(), g);
		
		Mat newPoints(maxPoints, 1, densePoints.type());
		Mat newColors(maxPoints, 1, denseColors.type());
		Mat newNormals;
		if (!denseNormals.empty())
			newNormals.create(maxPoints, 1, denseNormals.type());
			
		try {
			for(int i=0; i<maxPoints; ++i) {
				densePoints.row(indices[i]).copyTo(newPoints.row(i));
				denseColors.row(indices[i]).copyTo(newColors.row(i));
				if (!denseNormals.empty())
					denseNormals.row(indices[i]).copyTo(newNormals.row(i));
			}
		} catch (const cv::Exception& e) {
			std::cerr << "Error during subsampling: " << e.what() << std::endl;
			std::cerr << "densePoints: " << densePoints.size() << " rows=" << densePoints.rows << " cols=" << densePoints.cols << " type=" << densePoints.type() << std::endl;
			std::cerr << "denseColors: " << denseColors.size() << " rows=" << denseColors.rows << " cols=" << denseColors.cols << " type=" << denseColors.type() << std::endl;
			std::cerr << "denseNormals: " << denseNormals.size() << " rows=" << denseNormals.rows << " cols=" << denseNormals.cols << " type=" << denseNormals.type() << std::endl;
			std::cerr << "maxPoints: " << maxPoints << std::endl;
			std::cerr << "indices size: " << indices.size() << std::endl;
			throw;
		}
		
		densePoints = newPoints;
		denseColors = newColors;
		denseNormals = newNormals;
		
		std::cout << "number of subsampled points: " << densePoints.size() << std::endl;
	}

	// save dense point cloud
	makeDirectory(outputDir);
	viz::writeCloud(outputDir + "/point_cloud_dense.ply", densePoints, denseColors);

	// debug view
	if (debugView)
	{
		viz::Viz3d windowSparse("sparse point cloud (Press 'q' to show dense point cloud)");
		viz::Viz3d windowDense("dense point cloud (Press 'q' to exit)");
		drawReconstruction(windowSparse, cameras, rotations, translations, sparsePoints, sparseColors);
		drawReconstruction(windowDense, cameras, rotations, translations, densePoints, denseColors, denseNormals, normalLevel);
		windowSparse.spin();
		windowDense.spin();
	}

	return 0;
}
