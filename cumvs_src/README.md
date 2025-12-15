# cuda-multi-view-stereo

C++/CUDA library for Multi-View Stereo

---

## Description
cuda-multi-view-stereo (CUMVS) is a C++/CUDA library for Multi-View Stereo. This project currently supports depth map estimation using PatchMatch algorithms.

## Features

### Implementation of recent PatchMatch algorithms

This project aims to incorporate the essence of recent PatchMatch algorithms such as [ACMM](https://github.com/GhiXu/ACMM) while simplifying the implementation as much as possible.

### Performance optimization using GPU and CPU

The performance of CUDA kernels for propagation is optimized using shared memory, texture fetching, projective transformation approximations, etc. In addition, the performance of some processes performed on the CPU is optimized using multi-core parallelization and reduced disk access.

<div align="center">
<img src="https://github.com/fixstars/cuda-multi-view-stereo/wiki/images/runtime.png" width=1024>
</div>

### Support for debugging and visualization

The debug callback function allows you to check the status of the propagation during execution. In addition, you can instantly check the 3D reconstruction results using OpenCV's Viz module. Please check the [sample code](./samples/app_patch_match_mvs.cpp) for usage.

<div align="center">
<img src="https://github.com/fixstars/cuda-multi-view-stereo/wiki/images/demo.gif" width=1024><br/>
Visualization of propagation using debug callback (from left: reference image, depth map, normal map, cost map)
</div>

---

## Performance

### Settings
|Key|Value|
|---|---|
|CPU|Intel Core i7-13700K|
|GPU|NVIDIA GeForce RTX 3080|
|Maximum image size|3200 pixels for each dimension|

### Results on ETH3D High-res multi-view training dataset [2cm]

| method | Mean | courtyard | delivery_area | electro | facade | kicker | meadow | office | pipes | playground | relief | relief_2 | terrace | terrains |
|--------|------|-----------|---------------|---------|--------|--------|--------|--------|-------|------------|--------|----------|---------|----------|
| ACMM   | 78.9 | 87.5      | 83.7          | 86.8    | 70.8   | 77.5   | 66.4   | 64.9   | 70.1  | 72.1       | 85.4   | 84.9     | 89.9    | 85.5     |
| CUMVS  | 81.8 | 86.4      | 86.1          | 86.3    | 69.2   | 86.9   | 76.1   | 75.2   | 73    | 74.6       | 85.8   | 84.2     | 89.8    | 90.2     |

### Runtime (in second) on ETH3D High-res multi-view training dataset

| SCENE | Total   | courtyard | delivery_area | electro | facade | kicker | meadow | office | pipes | playground | relief | relief_2 | terrace | terrains |
|-------|---------|-----------|---------------|---------|--------|--------|--------|--------|-------|------------|--------|----------|---------|----------|
| ACMM  | 12277.7 | 1051.8    | 1014          | 1173.1  | 2471.1 | 908.6  | 337.2  | 595.7  | 233.5 | 1093.3     | 855.6  | 813.4    | 520.2   | 1210.2   |
| CUMVS | 2140.4  | 174.1     | 189.2         | 190.2   | 382.4  | 154.5  | 49.8   | 95.8   | 42.4  | 152.3      | 208.5  | 185.4    | 103.1   | 212.7    |

---

## Requirements
|Package Name|Minimum Requirements|Note|
|---|---|---|
|CMake|version >= 3.18||
|CUDA Toolkit|compute capability >= 6.0|
|OpenCV|version >= 4.6.0||
|OpenCV CUDA module|version >= 4.6.0|included in [opencv/opencv_contrib](https://github.com/opencv/opencv_contrib)|
|OpenCV Viz module|version >= 4.6.0|included in [opencv/opencv_contrib](https://github.com/opencv/opencv_contrib)|
|VTK|version >= 9.0|needed for OpenCV Viz module|

---

## How to build
```bash
$ git clone https://github.com/fixstars/cuda-multi-view-stereo.git
$ cd cuda-multi-view-stereo
$ mkdir build
$ cd build
$ cmake .. # specify CUDA architecture if necessary (e.g. -DCMAKE_CUDA_ARCHITECTURES=86)
$ make
```

### CMake options
|Option|Description|Default|
|---|---|---|
|CMAKE_CUDA_ARCHITECTURES|List of architectures to generate device code for|`52;61;72;75;86`|

## How to run
### samples

|Command|Description|
|---|---|
|`./samples/app_initialize_ETH3D scene-directory [options]`|Input data preparation for ETD3D dataset|
|`./samples/app_patch_match_mvs input-directory [options]`|Depth map estimation and 3D reconstruction|

Use the `--help` or `-h` option for detailed information.

### Example of running CUMVS on ETH3D High-res multi-view data

Here we will use the "pipes" scene as an example.
Download `pipes_dslr_undistorted.7z` from [ETH3D datasets](https://www.eth3d.net/datasets) and unzip it.

Create input data using `app_initialize_ETH3D`.

```bash
./samples/app_initialize_ETH3D /path_to_eth3d/pipes_dslr_undistorted/pipes --output-directory=inputs/pipes
```

Test `app_patch_match_mvs`.
The default image size is large and time consuming, so we recommend trying a smaller size first.
The option `-d` specifies debug display.

```bash
./samples/app_patch_match_mvs inputs/pipes --max-image-size=800 -d
```

If there are no problems, run the application with your preferred options.
After execution, a point cloud file `output/point_cloud_dense.ply` will be generated.

---

## Future work
- Improving depth estimation accuracy using plane prior and adaptive patch deformation
- Speeding up [ETH3D Multi-View Evaluation Program](https://github.com/ETH3D/multi-view-evaluation) and incorporate it into this project
- Implementing processes after mesh reconstruction

---

## Author
The "adaskit Team"

The adaskit is an open-source project created by [Fixstars Corporation](https://www.fixstars.com/) and its subsidiary companies including [Fixstars Autonomous Technologies](https://at.fixstars.com/), aimed at contributing to the ADAS industry by developing high-performance implementations for algorithms with high computational cost.

---

## License
Apache License 2.0
