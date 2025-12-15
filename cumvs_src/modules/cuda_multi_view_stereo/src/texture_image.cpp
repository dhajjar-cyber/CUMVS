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

#include "texture_image.h"

#include <cuda_runtime.h>
#include <iostream>

#include "cuda_macro.h"

namespace cv
{
namespace cuda
{

TextureImage::TextureImage() : textureObject_(0)
{
}

TextureImage::~TextureImage()
{
	release();
}

void TextureImage::create(const Mat& image, cudaTextureAddressMode addressMode, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode, Vec4f borderColor)
{
	CV_Assert(image.type() == CV_8U || image.type() == CV_32F);

	release();

	d_image_.upload(image);

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = d_image_.data;
	resDesc.res.pitch2D.height = d_image_.rows;
	resDesc.res.pitch2D.width = d_image_.cols;
	resDesc.res.pitch2D.pitchInBytes = d_image_.step;
	resDesc.res.pitch2D.desc = image.type() == CV_8U ? cudaCreateChannelDesc<uchar>() : cudaCreateChannelDesc<float>();

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.addressMode[0] = addressMode;
	texDesc.addressMode[1] = addressMode;
	texDesc.addressMode[2] = addressMode;
	texDesc.filterMode = filterMode;
	texDesc.readMode = readMode;
	texDesc.normalizedCoords = 0;
	texDesc.borderColor[0] = borderColor[0];
	texDesc.borderColor[1] = borderColor[1];
	texDesc.borderColor[2] = borderColor[2];
	texDesc.borderColor[3] = borderColor[3];

	CUDA_CHECK(cudaCreateTextureObject(&textureObject_, &resDesc, &texDesc, NULL));
}

void TextureImage::release()
{
	if (textureObject_)
	{
		CUDA_CHECK(cudaDestroyTextureObject(textureObject_));
		textureObject_ = 0;
	}
}

cudaTextureObject_t TextureImage::getTextureObject() const
{
	return textureObject_;
}

} // namespace cuda
} // namespace cv
