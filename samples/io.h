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

#ifndef __IO_H__
#define __IO_H__

#include <opencv2/core.hpp>

namespace cv
{

void makeDirectory(const String& path);

void saveImageInfo(const String& filename, const std::vector<String>& imageNames, InputArrayOfArrays cameraParams);
int loadImageInfo(const String& filename, std::vector<String>& imageNames, std::vector<Mat>& cameraParams);

void loadImages(const std::vector<String>& filenames, std::vector<Mat>& images, int flags);
void loadImagesAndCameras(const String& filenames, std::vector<Mat>& images, std::vector<Mat>& cameras, int maxImageSize = -1);

void saveGlobalPoses(const String& filename, InputArrayOfArrays Rs, InputArrayOfArrays ts);
void loadGlobalPoses(const String& filename, std::vector<Mat>& Rs, std::vector<Mat>& ts);

void saveViewIdSets(const String& filename, InputArrayOfArrays viewIdSets);
void loadViewIdSets(const String& filename, std::vector<Mat>& viewIdSets);

} // namespace cv

#endif // !__IO_H__
