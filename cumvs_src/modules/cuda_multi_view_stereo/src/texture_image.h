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

#ifndef __TEXTURE_IMAGE_H__
#define __TEXTURE_IMAGE_H__

#include <opencv2/core/cuda.hpp>
#include <texture_types.h>

namespace cv
{
namespace cuda
{

class TextureImage
{
public:

	TextureImage();
	~TextureImage();

	void create(const Mat& image, cudaTextureAddressMode addressMode = cudaAddressModeWrap, cudaTextureFilterMode filterMode = cudaFilterModePoint,
		cudaTextureReadMode readMode = cudaReadModeElementType, Vec4f borderColor = Vec4f(0, 0, 0, 1));
	void release();

	cudaTextureObject_t getTextureObject() const;

private:

	GpuMat d_image_;
	cudaTextureObject_t textureObject_;
};

} // namespace cuda
} // namespace cv

#endif // !__TEXTURE_IMAGE_H__
