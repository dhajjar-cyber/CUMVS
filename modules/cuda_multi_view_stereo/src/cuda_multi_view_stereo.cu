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

#include "cuda_multi_view_stereo_internal.h"

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_macro.h"

namespace cv
{
namespace cuda
{
namespace gpu
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum
{
	PIXEL_COLOR_BLK = 0,
	PIXEL_COLOR_RED = 1
};

enum ACMHPatterns
{
	FAR  = 0,
	NEAR = 3,
	DIAG = 6,
	FORE = 0,
	BACK = 1,
	ZERO = 2,
	TURN = 2,
	ODD  = 0,
	EVEN = 2,
};

static constexpr int PATTERN_08_DX[8] = { +0, +0, -5, -1, +1, +5, +0, +0 };
static constexpr int PATTERN_08_DY[8] = { -5, -1, +0, +0, +0, +0, +1, +5 };

static constexpr int MAX_VIEWS = 21;

static constexpr float INFINITY_COST = std::numeric_limits<float>::infinity();
static constexpr float GEOM_MAX_COST = 5.0f;
static constexpr float GEOM_COST_SCALE = 1 / GEOM_MAX_COST;
static constexpr float GEOM_COST_W = 0.5f;
static constexpr float PERTURBATION = 0.02f;
static constexpr float MAX_PATCH_COST = 2.f;
static constexpr float INVISIBLE_PATCH_COST = MAX_PATCH_COST + 1.f;

static constexpr int PATCH_SIZE = 9;
static constexpr int PATCH_RADIUS = PATCH_SIZE / 2;
static constexpr int SAMPLE_STEP = 2;
static_assert(PATCH_RADIUS% SAMPLE_STEP == 0, "PATCH_RADIUS must be multiples of SAMPLE_STEP");

static constexpr int PM_BLOCK_SIZE = 16;
static constexpr int SHARED_H = PM_BLOCK_SIZE + PATCH_SIZE - 1;
static constexpr int SHARED_W = 2 * PM_BLOCK_SIZE + PATCH_SIZE - 1;

static constexpr bool FIX_RANDOM_SEED = false;
static constexpr bool USE_NONLOCAL_STRATEGY = true;

static constexpr float PI_F = static_cast<float>(CV_PI);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Basic structs
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Point
template <typename T>
struct Point_
{
	__device__ inline Point_(T x = 0, T y = 0) : x(x), y(y) {}
	T x, y;
};

using Point2i = Point_<int>;
using Point2f = Point_<float>;
using Point = Point2i;

// Vec
template <typename T, int N>
struct Vec_
{
	__device__ inline Vec_() {}
	__device__ inline Vec_(T x) { val[0] = x; }
	__device__ inline Vec_(T x, T y) { val[0] = x; val[1] = y; }
	__device__ inline Vec_(T x, T y, T z) { val[0] = x; val[1] = y; val[2] = z; }

	__device__ inline T& operator()(int i) { return val[i]; }
	__device__ inline T operator()(int i) const { return val[i]; }
	__device__ inline T& operator[](int i) { return val[i]; }
	__device__ inline T operator[](int i) const { return val[i]; }

	__device__ inline Vec_ operator-() const
	{
		Vec_ vec;
#pragma unroll
		for (int i = 0; i < N; i++)
			vec.val[i] = -val[i];
		return vec;
	}

	__device__ inline T dot(const Vec_& rhs) const
	{
		const Vec_& lhs(*this);
		T sum = 0;
#pragma unroll
		for (int i = 0; i < N; i++)
			sum += lhs.val[i] * rhs.val[i];
		return sum;
	}

	T val[N];
};
using Vec3f = Vec_<float, 3>;

__device__ inline Vec3f operator*(const Vec3f& vec, float c) { return Vec3f(c * vec[0], c * vec[1], c * vec[2]); }
__device__ inline Vec3f operator*(float c, const Vec3f& vec) { return Vec3f(c * vec[0], c * vec[1], c * vec[2]); }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int ROWS, int COLS>
struct MatView
{
	__device__ inline T& operator()(int i, int j) { return data[j * ROWS + i]; }
	__device__ inline MatView(T* data) : data(data) {}
	T* data;
};

template <typename T, int ROWS, int COLS>
struct ConstMatView
{
	__device__ inline T operator()(int i, int j) const { return data[j * ROWS + i]; }
	__device__ inline ConstMatView(const T* data) : data(data) {}
	const T* data;
};

template <typename T, int ROWS, int COLS>
struct Matx
{
	using View = MatView<T, ROWS, COLS>;
	using ConstView = ConstMatView<T, ROWS, COLS>;
	__device__ inline T& operator()(int i, int j) { return data[i * COLS + j]; }
	__device__ inline T operator()(int i, int j) const { return data[i * COLS + j]; }
	__device__ inline operator View() { return View(data); }
	__device__ inline operator ConstView() const { return ConstView(data); }
	T data[ROWS * COLS];
};

using ConstMatView3x3d = ConstMatView<double, 3, 3>;
using Matx33f = Matx<float, 3, 3>;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TextureImage
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TextureImage
{
	__device__ inline TextureImage(const TexObjSz& texObjSz) : obj(texObjSz.obj), w(texObjSz.w), h(texObjSz.h) {}

	__device__ inline float operator()(float x, float y) const
	{
		return tex2D<float>(obj, x + 0.5f, y + 0.5f);
	}

	cudaTextureObject_t obj;
	int w, h;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SharedImage
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
using SharedPtrT = float(*)[SHARED_W];

struct SharedImage
{
	__device__ inline SharedImage(const TextureImage& image, SharedPtrT data, int blockw, int blockh) : data(data)
	{
		x0 = blockIdx.x * blockw - PATCH_RADIUS;
		y0 = blockIdx.y * blockh - PATCH_RADIUS;
		const int sharedw = blockw + PATCH_SIZE - 1;
		const int sharedh = blockh + PATCH_SIZE - 1;
		load(image, sharedw, sharedh);
	}

	__device__ inline void load(const TextureImage& image, int sharedw, int sharedh)
	{
		for (int dy = threadIdx.y; dy < sharedh; dy += blockDim.y)
			for (int dx = threadIdx.x; dx < sharedw; dx += blockDim.x)
				data[dy][dx] = image(x0 + dx, y0 + dy);
		__syncthreads();
	}

	__device__ inline float operator()(int x, int y) const
	{
		return data[y - y0][x - x0];
	}

	SharedPtrT data;
	int x0, y0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constant arrays
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__constant__ TexObjSz t_images[MAX_VIEWS];
__constant__ TexObjSz t_depths[MAX_VIEWS];
__constant__ Matx33f s_R21[MAX_VIEWS];
__constant__ Matx33f s_R12[MAX_VIEWS];
__constant__ Vec3f s_t21[MAX_VIEWS];
__constant__ Vec3f s_t12[MAX_VIEWS];
__constant__ float s_invK1[4]; // ifx1, ify1, icx1, icy1

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__host__ __device__ inline float floatCast(T v)
{
	return static_cast<float>(v);
}

__device__ inline float FMA(float x, float y, float z)
{
	return x * y + z;
}

template <class T>
__device__ inline T clamp(T x, T left, T right)
{
	return ::min(::max(x, left), right);
}

__device__ inline Vec3f composeNK(const Vec3f& n, float ifx, float ify, float icx, float icy)
{
	return Vec3f(ifx * n(0), ify * n(1), icx * n(0) + icy * n(1) + n(2));
}

__device__ inline Matx33f getHomographyMatrix(const Matx33f& R, const Vec3f& t, const Vec3f& n)
{
	Matx33f H;

	H(0, 0) = R(0, 0) - t(0) * n(0);
	H(0, 1) = R(0, 1) - t(0) * n(1);
	H(0, 2) = R(0, 2) - t(0) * n(2);

	H(1, 0) = R(1, 0) - t(1) * n(0);
	H(1, 1) = R(1, 1) - t(1) * n(1);
	H(1, 2) = R(1, 2) - t(1) * n(2);

	H(2, 0) = R(2, 0) - t(2) * n(0);
	H(2, 1) = R(2, 1) - t(2) * n(1);
	H(2, 2) = R(2, 2) - t(2) * n(2);

	return H;
}

__device__ inline Point2f homographyTransform(float x, float y, const Matx33f& H)
{
	const float X = H(0, 0) * x + H(0, 1) * y + H(0, 2);
	const float Y = H(1, 0) * x + H(1, 1) * y + H(1, 2);
	const float Z = H(2, 0) * x + H(2, 1) * y + H(2, 2);
	const float invZ = 1 / Z;

	return Point2f(invZ * X, invZ * Y);
}

__device__ inline Vec3f rigidTransform(const Matx33f& R, const Vec3f& x, const Vec3f& t)
{
	Vec3f y;
	y(0) = FMA(R(0, 0), x(0), FMA(R(0, 1), x(1), FMA(R(0, 2), x(2), t(0))));
	y(1) = FMA(R(1, 0), x(0), FMA(R(1, 1), x(1), FMA(R(1, 2), x(2), t(1))));
	y(2) = FMA(R(2, 0), x(0), FMA(R(2, 1), x(1), FMA(R(2, 2), x(2), t(2))));
	return y;
}

__device__ inline void partialSelectionSort(float arr[], int len, int nSelecton)
{
	for (int j = 0; j < nSelecton; j++)
	{
		int minIdx = j;
		for (int k = j + 1; k < len; k++)
		{
			if (arr[minIdx] > arr[k])
				minIdx = k;
		}

		float temp;
		temp = arr[minIdx];
		arr[minIdx] = arr[j];
		arr[j] = temp;
	}
}

__device__ inline unsigned int getRandomSeed(int x, int y, int w)
{
	unsigned int seed = y * w + x;
	if constexpr (!FIX_RANDOM_SEED)
		seed = seed * clock() + 1;
	return seed;
}

__device__ inline Vec3f normalize(int x, int y)
{
	return Vec3f(s_invK1[0] * x + s_invK1[2], s_invK1[1] * y + s_invK1[3], 1);
}

__device__ inline float interpZ(int x1, int y1, int x2, int y2, float Z2, Vec3f N)
{
	const auto Z1 = Z2 * N.dot(normalize(x2, y2)) / N.dot(normalize(x1, y1));
	return Z1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ZNCC
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define HOMOGRAPHY_APPROXIMATION

struct WeightedZNCC
{
	__device__ inline WeightedZNCC() : sIT(0), sII(0), sTT(0), sI(0), sT(0), sw(0)
	{
	}

	__device__ inline void add(float I, float T, float w)
	{
		sIT += w * I * T;
		sII += w * I * I;
		sTT += w * T * T;
		sI += w * I;
		sT += w * T;
		sw += w;
	}

	__device__ inline float compute()
	{
		constexpr float MIN_VAR = 1e-6f;
		const float invsw = 1.f / sw;

		sIT *= invsw;
		sII *= invsw;
		sTT *= invsw;
		sI *= invsw;
		sT *= invsw;

		const float DI = sII - sI * sI;
		const float DT = sTT - sT * sT;

		if (DI <= MIN_VAR || DT <= MIN_VAR)
			return -1;

		return (sIT - sI * sT) / sqrtf(DI * DT);
	}

	float sIT, sII, sTT, sI, sT, sw;
};

__device__ inline float calcBilateralWeight(float dx, float dy, float dI)
{
	constexpr float SIGMA_SPATIAL = PATCH_RADIUS;
	constexpr float SIGMA_COLOR = 0.2f;

	constexpr float ALPHA = -1.f / (2 * SIGMA_SPATIAL * SIGMA_SPATIAL);
	constexpr float BETA = -1.f / (2 * SIGMA_COLOR * SIGMA_COLOR);

	const float spatialDist = dx * dx + dy * dy;
	const float colorDist = dI * dI;

	return expf(ALPHA * spatialDist + BETA * colorDist);
}

__device__ inline bool isVisibleTransform(const Matx33f& H, int x, int y, int w, int h)
{
	const float X = H(0, 0) * x + H(0, 1) * y + H(0, 2);
	const float Y = H(1, 0) * x + H(1, 1) * y + H(1, 2);
	const float Z = H(2, 0) * x + H(2, 1) * y + H(2, 2);
	return X >= 0 && X < Z * w && Y >= 0 && Y < Z * h;
}

__device__ inline float calcBilateralZNCC(const Matx33f& H, const SharedImage& I1, const TextureImage& I2, int cx, int cy)
{
	if (!isVisibleTransform(H, cx, cy, I2.w, I2.h))
		return INVISIBLE_PATCH_COST;

	WeightedZNCC ZNCC;

	const float Ic = I1(cx, cy);

#ifdef HOMOGRAPHY_APPROXIMATION

	for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy += SAMPLE_STEP)
	{
		const int y1 = cy + dy;

		// first-order approximation around cx
		const float U = H(0, 0) * cx + H(0, 1) * y1 + H(0, 2);
		const float V = H(1, 0) * cx + H(1, 1) * y1 + H(1, 2);
		const float W = H(2, 0) * cx + H(2, 1) * y1 + H(2, 2);
		const float invW = 1.f / W;

		const float ax = invW * invW * (H(0, 0) * W - U * H(2, 0));
		const float bx = invW * U;

		const float ay = invW * invW * (H(1, 0) * W - V * H(2, 0));
		const float by = invW * V;

		for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx += SAMPLE_STEP)
		{
			const int x1 = cx + dx;
			const float x2 = ax * dx + bx;
			const float y2 = ay * dx + by;

			const float I = I1(x1, y1);
			const float T = I2(x2, y2);
			const float w = calcBilateralWeight(dx, dy, I - Ic);

			ZNCC.add(I, T, w);
		}
	}

#else

	for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy += SAMPLE_STEP)
	{
		for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx += SAMPLE_STEP)
		{
			const int x1 = cx + dx;
			const int y1 = cy + dy;
			const Point2f pt2 = homographyTransform(x1, y1, H);

			const float I = I1(x1, y1);
			const float T = I2(pt2.x, pt2.y);
			const float w = calcBilateralWeight(dx, dy, I - Ic);

			ZNCC.add(I, T, w);
		}
	}

#endif // !HOMOGRAPHY_APPROXIMATION

	return ::max(1.f - ZNCC.compute(), 0.f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Matching Cost
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ inline float calcGeomConsistencyCost(int x1, int y1, float Z1, int idx)
{
	const auto D2 = TextureImage(t_depths[idx + 1]);

	// project into neighbor camera coordinate
	const auto Xc2 = rigidTransform(s_R21[idx], Vec3f(Z1 * x1, Z1 * y1, Z1), s_t21[idx]);
	const auto Z2 = Xc2(2);
	if (Z2 <= 0)
		return INVISIBLE_PATCH_COST;

	const auto invZ2 = 1.f / Z2;
	const auto x2 = invZ2 * Xc2(0);
	const auto y2 = invZ2 * Xc2(1);

	// again, reproject to reference camera coordinate
	const auto _Z2 = D2(x2, y2);
	if (_Z2 <= 0)
		return INVISIBLE_PATCH_COST;

	const auto _Xc1 = rigidTransform(s_R12[idx], Vec3f(_Z2 * x2, _Z2 * y2, _Z2), s_t12[idx]);
	const auto _Z1 = _Xc1(2);

	const auto _invZ1 = 1.f / _Z1;
	const auto _x1 = _invZ1 * _Xc1(0);
	const auto _y1 = _invZ1 * _Xc1(1);

	const float cost = ::hypotf(x1 - _x1, y1 - _y1);
	return GEOM_COST_SCALE * ::min(cost, GEOM_MAX_COST);
}

__device__ inline float calcGeomConsistencyCost(int cx, int cy, const Vec3f& m, int idx)
{
	// const auto D1 = t_depths[0];
	const auto D2 = TextureImage(t_depths[idx + 1]);
	const auto H = getHomographyMatrix(s_R21[idx], s_t21[idx], m);

	constexpr int GEOM_STEP = 2;
	constexpr int GEOM_RADIUS = 2;

	float sum = 0;
	int cnt = 0;
	for (int dy = -GEOM_RADIUS; dy <= GEOM_RADIUS; dy += GEOM_STEP)
	{
		for (int dx = -GEOM_RADIUS; dx <= GEOM_RADIUS; dx += GEOM_STEP)
		{
			const int x1 = cx + dx;
			const int y1 = cy + dy;
			const Point2f pt2 = homographyTransform(x1, y1, H);
			const float x2 = pt2.x;
			const float y2 = pt2.y;
			const auto _Z2 = D2(x2, y2);
			if (_Z2 > 0)
			{
				const auto _Xc1 = rigidTransform(s_R12[idx], Vec3f(_Z2 * x2, _Z2 * y2, _Z2), s_t12[idx]);
				const auto _Z1 = _Xc1(2);

				const auto _invZ1 = 1.f / _Z1;
				const auto _x1 = _invZ1 * _Xc1(0);
				const auto _y1 = _invZ1 * _Xc1(1);

				const float dist = ::hypotf(x1 - _x1, y1 - _y1);
				const float cost = dist;
				sum += cost;
				cnt++;
			}
		}
	}

	if (cnt == 0)
		return GEOM_MAX_COST;

	return GEOM_COST_SCALE * ::min(sum / cnt, GEOM_MAX_COST);
}

__device__ inline Vec3f calcNormalParams(int x1, int y1, float Z1, Vec3f N1)
{
	const auto p = Vec3f(Z1 * x1, Z1 * y1, Z1);
	const auto n = composeNK(N1, s_invK1[0], s_invK1[1], s_invK1[2], s_invK1[3]);
	const auto d = -n.dot(p);
	const auto m = (1 / d) * n;
	return m;
}

__device__ inline float calcPatchCost(const SharedImage& I1, int x1, int y1, const Vec3f& m, int idx)
{
	const auto I2 = t_images[idx + 1];
	const auto H = getHomographyMatrix(s_R21[idx], s_t21[idx], m);
	const auto cost = calcBilateralZNCC(H, I1, I2, x1, y1);
	return cost;
}

__device__ inline float combineCosts(float C1, float C2)
{
	return (1.f - GEOM_COST_W) * C1 + GEOM_COST_W * C2;
}

__device__ inline float calcPatchCostWithGC(const SharedImage& I1, int x1, int y1, const Vec3f& m, int idx)
{
	const float C1 = calcPatchCost(I1, x1, y1, m, idx);
	const float C2 = calcGeomConsistencyCost(x1, y1, m, idx);
	return C1 <= MAX_PATCH_COST ? combineCosts(C1, C2) : INVISIBLE_PATCH_COST;
}

template <class CostFunc>
__device__ inline float aggregateCosts(CostFunc&& costFunc, int nothers)
{
	float sumCost = 0;
	int count = 0;
	for (int i = 0; i < nothers; i++)
	{
		const auto cost = costFunc(i);
		if (cost <= MAX_PATCH_COST)
		{
			sumCost += cost;
			count++;
		}
	}
	return count > 0 ? sumCost / count : MAX_PATCH_COST;
}

template <class CostFunc>
__device__ inline float aggregateCosts(CostFunc&& costFunc, int nothers, const float* weights)
{
	float sumCost = 0;
	for (int i = 0; i < nothers; i++)
		sumCost += weights[i] * costFunc(i);
	return sumCost;
}

struct MatchingCost
{
	__device__ MatchingCost(const SharedImage& I1, int nviews, bool geom) : I1(I1), nothers(nviews - 1), geom(geom)
	{
	}

	__device__ inline float compute(int x1, int y1, float Z1, Vec3f N1) const
	{
		const auto m = calcNormalParams(x1, y1, Z1, N1);
		return geom ?
			aggregateCosts([&](int i) { return calcPatchCostWithGC(I1, x1, y1, m, i); }, nothers) :
			aggregateCosts([&](int i) { return calcPatchCost(I1, x1, y1, m, i); }, nothers);
	}

	__device__ inline float compute(int x1, int y1, float Z1, Vec3f N1, const float* weights) const
	{
		const auto m = calcNormalParams(x1, y1, Z1, N1);
		return geom ?
			aggregateCosts([&](int i) { return calcPatchCostWithGC(I1, x1, y1, m, i); }, nothers, weights) :
			aggregateCosts([&](int i) { return calcPatchCost(I1, x1, y1, m, i); }, nothers, weights);
	}

	__device__ inline float compute(int x1, int y1, float Z1, const float* patchCosts, const float* weights) const
	{
		return geom ?
			aggregateCosts([&](int i) { return combineCosts(patchCosts[i], calcGeomConsistencyCost(x1, y1, Z1, i)); }, nothers, weights) :
			aggregateCosts([&](int i) { return patchCosts[i]; }, nothers, weights);
	}

	const SharedImage& I1;
	int nothers;
	bool geom;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Propagate
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <int DX, int DY>
__device__ inline void update_(const MatchingCost& MC, const float* Z, const Vec3f* N,
	int w, int h, int cx, int cy, float& minC, Point& minP)
{
	const int nx = cx + DX;
	const int ny = cy + DY;
	if (nx >= 0 && nx < w && ny >= 0 && ny < h)
	{
		const float newC = MC.compute(cx, cy, Z[ny * w + nx], N[ny * w + nx]);
		if (newC < minC)
		{
			minC = newC;
			minP = Point(nx, ny);
		}
	}
}

__device__ inline void update08(const MatchingCost& MC, const float* Z, const Vec3f* N,
	int w, int h, int x, int y, float& minC, Point& minP)
{
	update_<PATTERN_08_DX[0], PATTERN_08_DY[0]>(MC, Z, N, w, h, x, y, minC, minP);
	update_<PATTERN_08_DX[1], PATTERN_08_DY[1]>(MC, Z, N, w, h, x, y, minC, minP);
	update_<PATTERN_08_DX[2], PATTERN_08_DY[2]>(MC, Z, N, w, h, x, y, minC, minP);
	update_<PATTERN_08_DX[3], PATTERN_08_DY[3]>(MC, Z, N, w, h, x, y, minC, minP);
	update_<PATTERN_08_DX[4], PATTERN_08_DY[4]>(MC, Z, N, w, h, x, y, minC, minP);
	update_<PATTERN_08_DX[5], PATTERN_08_DY[5]>(MC, Z, N, w, h, x, y, minC, minP);
	update_<PATTERN_08_DX[6], PATTERN_08_DY[6]>(MC, Z, N, w, h, x, y, minC, minP);
	update_<PATTERN_08_DX[7], PATTERN_08_DY[7]>(MC, Z, N, w, h, x, y, minC, minP);
}

__device__ inline void propagateGipuma(const MatchingCost& MC, float* Z, Vec3f* N, float* C,
	int w, int h, int x, int y)
{
	const float iniC = C[y * w + x];
	float minC = iniC;
	Point minP(x, y);

	update08(MC, Z, N, w, h, x, y, minC, minP);
	// update
	if (minC < iniC)
	{
		Z[y * w + x] = Z[minP.y * w + minP.x];
		N[y * w + x] = N[minP.y * w + minP.x];
		C[y * w + x] = minC;
	}
}

template <int DIST_ID, int DX_ID, int DY_ID, int N, int OFFSET = 0>
__device__ inline Point findSamplingPointACMH(const float* C, int w, int h, int cx, int cy)
{
	constexpr int ACMH_PATTERNS[10][11] =
	{
		{ +3, +5, +7, +9, +11, +13, +15, +17, +19, +21, +23 }, // FAR FORE
		{ -3, -5, -7, -9, -11, -13, -15, -17, -19, -21, -23 }, // FAR BACK
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },                   // FAR ZERO
		{ +1, +2, +2, +3, +3, +4, +4, +5, +5, +6, +6 },        // NEAR FORE
		{ -1, -2, -2, -3, -3, -4, -4, -5, -5, -6, -6 },        // NEAR BACK
		{ +0, -1, +1, -2, +2, -3, +3, -4, +4, -5, +5 },        // NEAR TURN
		{ +5, +7, +7, +9, +9, +11, +11, +13, +13, +15, +15 },  // DIAG FORE ODD
		{ -5, -7, -7, -9, -9, -11, -11, -13, -13, -15, -15 },  // DIAG BACk ODD
		{ +6, +6, +8, +8, +10, +10, +12, +12, +14, +14, +16 }, // DIAG FORE EVEN
		{ -6, -6, -8, -8, -10, -10, -12, -12, -14, -14, -16 }  // DIAG BACk EVEN
	};

	constexpr int PATTERN_X = DIST_ID + DX_ID;
	constexpr int PATTERN_Y = DIST_ID + DY_ID;

	float minC = INFINITY_COST;
	Point minP(-1, -1);
	for (int i = 0; i < N; i++)
	{
		const int nx = cx + ACMH_PATTERNS[PATTERN_X][i + OFFSET];
		const int ny = cy + ACMH_PATTERNS[PATTERN_Y][i + OFFSET];
		if (nx >= 0 && nx < w && ny >= 0 && ny < h)
		{
			const float newC = C[ny * w + nx];
			if (newC < minC)
			{
				minC = newC;
				minP = Point(nx, ny);
			}
		}
	}
	return minP;
}

__device__ inline void findSamplingPoints(const float* C, int w, int h, int x, int y, Point samplingPoints[8])
{
	if constexpr (USE_NONLOCAL_STRATEGY)
	{
		samplingPoints[0] = findSamplingPointACMH<FAR, ZERO, BACK, 5, 1>(C, w, h, x, y); // U Far
		samplingPoints[1] = findSamplingPointACMH<FAR, ZERO, FORE, 5, 1>(C, w, h, x, y); // D Far
		samplingPoints[2] = findSamplingPointACMH<FAR, BACK, ZERO, 5, 1>(C, w, h, x, y); // L Far
		samplingPoints[3] = findSamplingPointACMH<FAR, FORE, ZERO, 5, 1>(C, w, h, x, y); // R Far
		samplingPoints[4] = findSamplingPointACMH<DIAG, BACK + ODD, BACK + EVEN, 8>(C, w, h, x, y); // L U
		samplingPoints[5] = findSamplingPointACMH<DIAG, FORE + EVEN, BACK + ODD, 8>(C, w, h, x, y); // R U
		samplingPoints[6] = findSamplingPointACMH<DIAG, BACK + EVEN, FORE + ODD, 8>(C, w, h, x, y); // L D
		samplingPoints[7] = findSamplingPointACMH<DIAG, FORE + ODD, FORE + EVEN, 8>(C, w, h, x, y); // R D
	}
	else
	{
		samplingPoints[0] = findSamplingPointACMH<FAR, ZERO, BACK, 11>(C, w, h, x, y); // U Far
		samplingPoints[1] = findSamplingPointACMH<FAR, ZERO, FORE, 11>(C, w, h, x, y); // D Far
		samplingPoints[2] = findSamplingPointACMH<FAR, BACK, ZERO, 11>(C, w, h, x, y); // L Far
		samplingPoints[3] = findSamplingPointACMH<FAR, FORE, ZERO, 11>(C, w, h, x, y); // R Far
		samplingPoints[4] = findSamplingPointACMH<NEAR, TURN, BACK, 5>(C, w, h, x, y); // U Near
		samplingPoints[5] = findSamplingPointACMH<NEAR, TURN, FORE, 5>(C, w, h, x, y); // D Near
		samplingPoints[6] = findSamplingPointACMH<NEAR, BACK, TURN, 5>(C, w, h, x, y); // L Near
		samplingPoints[7] = findSamplingPointACMH<NEAR, FORE, TURN, 5>(C, w, h, x, y); // R Near
	}
}

__device__ inline void propagateACMH(const MatchingCost& MC, float* Z, Vec3f* N, float* C,
	int w, int h, int x, int y, float minDepth, float maxDepth, float* viewWeights)
{
	float costMat[8][MAX_VIEWS];
	const int nothers = MC.nothers;

	Point samplingPoints[8];
	findSamplingPoints(C, w, h, x, y, samplingPoints);

	for (int areaId = 0; areaId < 8; areaId++)
	{
		const Point p = samplingPoints[areaId];
		if (p.x >= 0 && p.y >= 0)
		{
			const auto n = calcNormalParams(p.x, p.y, Z[p.y * w + p.x], N[p.y * w + p.x]);
			for (int viewId = 0; viewId < nothers; viewId++)
				costMat[areaId][viewId] = calcPatchCost(MC.I1, x, y, n, viewId);
		}
		else
		{
			for (int viewId = 0; viewId < nothers; viewId++)
				costMat[areaId][viewId] = INVISIBLE_PATCH_COST;
		}
	}

	float weightSum = 0;
	for (int viewId = 0; viewId < nothers; viewId++)
	{
		float costSum = 0;
		for (int areaId = 0; areaId < 8; areaId++)
			costSum += costMat[areaId][viewId];

		const float weight = ::max(1.f - (1.f / 8) * costSum, 0.f);
		viewWeights[viewId] = weight;
		weightSum += weight;
	}

	const float scale = 1.f / weightSum;
	for (int viewId = 0; viewId < nothers; viewId++)
		viewWeights[viewId] = weightSum > 0 ? scale * viewWeights[viewId] : 1.f / nothers;

	float minZ = -1.f;
	float minC = INFINITY_COST;
	Point minP = samplingPoints[0];
	for (int areaId = 0; areaId < 8; areaId++)
	{
		const auto& p = samplingPoints[areaId];
		if (p.x >= 0 && p.y >= 0)
		{
			const float newZ = interpZ(x, y, p.x, p.y, Z[p.y * w + p.x], N[p.y * w + p.x]);
			const float newC = MC.compute(x, y, newZ, costMat[areaId], viewWeights);
			if (newC < minC)
			{
				minZ = newZ;
				minC = newC;
				minP = p;
			}
		}
	}

	C[y * w + x] = MC.compute(x, y, Z[y * w + x], N[y * w + x], viewWeights);
	if (minC < C[y * w + x] && minZ >= minDepth && minZ <= maxDepth)
	{
		Z[y * w + x] = minZ;
		N[y * w + x] = N[minP.y * w + minP.x];
		C[y * w + x] = minC;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Refine
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Xorshift
{
	__device__ inline Xorshift(unsigned int seed) : val(seed) {}

	__device__ inline int uniform(int left, int right)
	{
		generate();
		return val % (right - left + 1) + left;
	}

	__device__ inline float uniform(float left, float right)
	{
		constexpr float SCALE = static_cast<float>(1. / UINT_MAX);
		generate();
		return SCALE * val * (right - left) + left;
	}

	__device__ inline void generate()
	{
		val ^= (val << 13);
		val ^= (val >> 17);
		val ^= (val << 5);
	}

	unsigned int val;
};

__device__ inline float generateRandomDepth(Xorshift& random, float minZ, float maxZ)
{
	return random.uniform(minZ, maxZ);
}

__device__ inline float generatePerturbedDepth(Xorshift& random, float center, float delta, float minZ, float maxZ)
{
	return clamp(center + random.uniform(-delta, +delta), minZ, maxZ);
}

__device__ inline Vec3f generateNormal(float theta, float phi)
{
	const float sinTheta = sinf(theta);
	const float cosTheta = cosf(theta);
	const float sinPhi = sinf(phi);
	const float cosPhi = cosf(phi);

	return Vec3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

__device__ inline Vec3f generateRandomNormal(int x, int y, Xorshift& random)
{
	const float theta = random.uniform(0.f, PI_F);
	const float phi = random.uniform(0.f, PI_F * 2.f);

	const auto N = generateNormal(theta, phi);
	return N.dot(normalize(x, y)) <= 0.f ? N : -N;
}

__device__ inline Vec3f generatePerturbedNormal(int x, int y, Xorshift& random, Vec3f center, float delta)
{
	const float X = center(0);
	const float Y = center(1);
	const float Z = center(2);
	
	const float pertubedTheta = acosf(Z) + random.uniform(-delta, +delta);
	const float pertubedPhi = atan2f(Y, X) + random.uniform(-delta, +delta);

	const auto pertubed = generateNormal(pertubedTheta, pertubedPhi);
	return pertubed.dot(normalize(x, y)) <= 0.f ? pertubed : center;
}

__device__ inline void refineGipuma(const MatchingCost& MC, float* Z, Vec3f* N, float* C, int w, int h, int x, int y,
	float deltaDepth, float deltaAngle, float minDepth, float maxDepth, int niterations, Xorshift& random)
{
	const float iniC = C[y * w + x];
	float minC = iniC;
	float minZ = Z[y * w + x];
	Vec3f minN = N[y * w + x];

	for (int i = 0; i < niterations; i++)
	{
		const float newZ = generatePerturbedDepth(random, minZ, deltaDepth, minDepth, maxDepth);
		const Vec3f newN = generatePerturbedNormal(x, y, random, minN, deltaAngle);
		const float newC = MC.compute(x, y, newZ, newN);
		if (newC < minC)
		{
			minZ = newZ;
			minN = newN;
			minC = newC;
		}
	}
	if (minC < iniC)
	{
		Z[y * w + x] = minZ;
		N[y * w + x] = minN;
		C[y * w + x] = minC;
	}
}

__device__ inline void refineACMH(const MatchingCost& MC, float* Z, Vec3f* N, float* C, int w, int h, int x, int y,
	float minDepth, float maxDepth, const float* viewWeights, Xorshift& random)
{

	const float curZ = Z[y * w + x];
	const Vec3f curN = N[y * w + x];

	const float randZ = generateRandomDepth(random, minDepth, maxDepth);
	const float pertZ = generatePerturbedDepth(random, curZ, curZ * PERTURBATION, minDepth, maxDepth);

	const Vec3f randN = generateRandomNormal(x, y, random);
	const Vec3f pertN = generatePerturbedNormal(x, y, random, curN, PERTURBATION * 0.5f * PI_F);

	// refine candidates
	const float candsZ[5] = { randZ, curZ,  randZ, curZ,  pertZ };
	const Vec3f candsN[5] = { curN,  randN, randN, pertN, curN };

	const float iniC = C[y * w + x];
	float minC = iniC;
	float minZ = Z[y * w + x];
	Vec3f minN = N[y * w + x];

	for (int i = 0; i < 5; i++)
	{
		float newZ = candsZ[i];
		Vec3f newN = candsN[i];
		float newC = MC.compute(x, y, newZ, newN, viewWeights);

		if (newC < minC)
		{
			minZ = newZ;
			minN = newN;
			minC = newC;
		}
	}

	// update
	if (minC < iniC)
	{
		Z[y * w + x] = minZ;
		N[y * w + x] = minN;
		C[y * w + x] = minC;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void propagateAndRefineGipumaKernel(float* Z, Vec3f* N, float* C, int w, int h, int nviews,
	int color, float minZ, float maxZ, float deltaDepth, float deltaAngle, int niterations, bool geom)
{
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + ((y % 2) ^ color);

	__shared__ float buffer[SHARED_H][SHARED_W];
	SharedImage I1(t_images[0], buffer, 2 * blockDim.x, blockDim.y);

	if (x >= w || y >= h)
		return;

	MatchingCost MC(I1, nviews, geom);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// propagate
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	propagateGipuma(MC, Z, N, C, w, h, x, y);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// refine
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	Xorshift random(getRandomSeed(x, y, w));
	refineGipuma(MC, Z, N, C, w, h, x, y, deltaDepth, deltaAngle, minZ, maxZ, niterations, random);
}

__global__ void propagateAndRefineACMHKernel(float* Z, Vec3f* N, float* C, int w, int h, int nviews,
	int color, float minZ, float maxZ, bool geom)
{
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + ((y % 2) ^ color);

	__shared__ float buffer[SHARED_H][SHARED_W];
	SharedImage I1(t_images[0], buffer, 2 * blockDim.x, blockDim.y);

	if (x >= w || y >= h)
		return;

	MatchingCost MC(I1, nviews, geom);
	float viewWeights[MAX_VIEWS];

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// propagate
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	propagateACMH(MC, Z, N, C, w, h, x, y, minZ, maxZ, viewWeights);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// refine
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	Xorshift random(getRandomSeed(x, y, w));
	refineACMH(MC, Z, N, C, w, h, x, y, minZ, maxZ, viewWeights, random);
}

__global__ void calcInitialCostsKernel(const PtrStepSzf depths, const PtrStepSz<Vec3f> normals, PtrStepSzf costs, int nviews, bool geom)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float buffer[SHARED_H][SHARED_W];
	SharedImage I1(t_images[0], buffer, blockDim.x, blockDim.y);

	if (x >= depths.cols || y >= depths.rows)
		return;

	MatchingCost MC(I1, nviews, geom);
	costs(y, x) = MC.compute(x, y, depths(y, x), normals(y, x));
}

__global__ void initializeDepthsAndNormalsKernel(PtrStepSzf depths, PtrStepSz<Vec3f> normals, float minZ, float maxZ)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= depths.cols || y >= depths.rows)
		return;

	Xorshift random(getRandomSeed(x, y, normals.cols));
	depths(y, x) = generateRandomDepth(random, minZ, maxZ);
	normals(y, x) = generateRandomNormal(x, y, random);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Public functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void waitForKernelCompletion()
{
	CUDA_CHECK(cudaDeviceSynchronize());
}

void uploadImages(const std::vector<TexObjSz>& images)
{
	CUDA_CHECK(cudaMemcpyToSymbol(t_images, images.data(), sizeof(TexObjSz) * images.size()));
}

void uploadDepths(const std::vector<TexObjSz>& depths)
{
	CUDA_CHECK(cudaMemcpyToSymbol(t_depths, depths.data(), sizeof(TexObjSz) * depths.size()));
}

void uploadHomographyParams(const cv::Vec4f& invK1, const std::vector<cv::Matx33f>& R21, const std::vector<cv::Vec3f>& t21)
{
	CUDA_CHECK(cudaMemcpyToSymbol(s_invK1, invK1.val, sizeof(float) * 4));
	CUDA_CHECK(cudaMemcpyToSymbol(s_R21, R21.data(), sizeof(cv::Matx33f) * R21.size()));
	CUDA_CHECK(cudaMemcpyToSymbol(s_t21, t21.data(), sizeof(cv::Vec3f) * t21.size()));
}

void uploadHomographyParams(const cv::Vec4f& invK1, const std::vector<cv::Matx33f>& R21, const std::vector<cv::Vec3f>& t21,
	const std::vector<cv::Matx33f>& R12, const std::vector<cv::Vec3f>& t12)
{
	CUDA_CHECK(cudaMemcpyToSymbol(s_invK1, invK1.val, sizeof(float) * 4));
	CUDA_CHECK(cudaMemcpyToSymbol(s_R21, R21.data(), sizeof(cv::Matx33f) * R21.size()));
	CUDA_CHECK(cudaMemcpyToSymbol(s_t21, t21.data(), sizeof(cv::Vec3f) * t21.size()));
	CUDA_CHECK(cudaMemcpyToSymbol(s_R12, R12.data(), sizeof(cv::Matx33f) * R12.size()));
	CUDA_CHECK(cudaMemcpyToSymbol(s_t12, t12.data(), sizeof(cv::Vec3f) * t12.size()));
}

void propagateAndRefineGipuma(GpuMat& depths, GpuMat& normals, GpuMat& costs, int nviews, int color,
	float minZ, float maxZ, float deltaDepth, float deltaAngle, int niterations, bool geom)
{
	const int h = depths.rows;
	const int w = depths.cols;

	const int npixelsY = h;
	const int npixelsX = w / 2;

	constexpr int BLOCK_SIZE = PM_BLOCK_SIZE;
	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	const dim3 grid(divUp(npixelsX, block.x), divUp(npixelsY, block.y));

	auto Z = depths.ptr<float>();
	auto N = normals.ptr<Vec3f>();
	auto C = costs.ptr<float>();

	propagateAndRefineGipumaKernel<<<grid, block>>>(Z, N, C, w, h, nviews, color, minZ, maxZ, deltaDepth, deltaAngle, niterations, geom);
	CUDA_CHECK(cudaGetLastError());
}

void propagateAndRefineACMH(GpuMat& depths, GpuMat& normals, GpuMat& costs, int nviews, int color,
	float minZ, float maxZ, bool geom)
{
	const int h = depths.rows;
	const int w = depths.cols;

	const int npixelsY = h;
	const int npixelsX = w / 2;

	constexpr int BLOCK_SIZE = PM_BLOCK_SIZE;
	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	const dim3 grid(divUp(npixelsX, block.x), divUp(npixelsY, block.y));

	auto Z = depths.ptr<float>();
	auto N = normals.ptr<Vec3f>();
	auto C = costs.ptr<float>();

	propagateAndRefineACMHKernel<<<grid, block>>>(Z, N, C, w, h, nviews, color, minZ, maxZ, geom);
	CUDA_CHECK(cudaGetLastError());
}

void calcInitialCosts(const GpuMat& depths, const GpuMat& normals, GpuMat& costs, int nviews, bool geom)
{
	const int h = normals.rows;
	const int w = normals.cols;

	constexpr int BLOCK_SIZE = PM_BLOCK_SIZE;
	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	const dim3 grid(divUp(w, block.x), divUp(h, block.y));

	calcInitialCostsKernel<<<grid, block>>>(depths, normals, costs, nviews, geom);
	CUDA_CHECK(cudaGetLastError());
}

void initializeDepthsAndNormals(GpuMat& depths, GpuMat& normals, float minZ, float maxZ)
{
	const int w = normals.cols;
	const int h = normals.rows;

	constexpr int BLOCK_SIZE = 16;
	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	const dim3 grid(divUp(w, block.x), divUp(h, block.y));

	initializeDepthsAndNormalsKernel<<<grid, block>>>(depths, normals, minZ, maxZ);
	CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace cuda
} // namespace cv
