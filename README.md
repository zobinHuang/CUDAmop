# CUDAop (CUDA Operators and Profiling)

This repo provides implementations of common CUDA opeartors and corresponding profiling-program suite, including:

* vector addition
    * Manually-manaed Memory
    * Unified Memory
    * Unified Memory with Prefetching

I also wrote corresponding blogs (in Chinses) for all the underhood details behind these profiling test (available [here](https://zobinhuang.github.io/sec_learning/Tech_OS_And_Linux_Kernel/index.html#cuda)), welcome to read and comments if you have any suggestion.

## Build Project

### Preparation
1. Host equipped with NVIDIA CUDA-capable GPU, see [CUDA GPUs - NVIDIA Developer](https://developer.nvidia.com/cuda-gpus);
2. OS with NVIDIA Driver and CUDA Tookit installed, to check:

```bash
# check driver status
$ nvidia-smi
Wed Jul 27 13:54:53 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.129.06   Driver Version: 470.129.06   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA RTX A4000    Off  | 00000000:06:10.0 Off |                  Off |
| 41%   42C    P8    17W / 140W |     10MiB / 16117MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1137      G   /usr/lib/xorg/Xorg                  4MiB |
|    0   N/A  N/A    134697      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+

# check cuda status
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_18:49:52_PDT_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
```

3. OS with build essential tools installed

```bash
# Ubuntu
sudo apt-get install build-essential

# CentOS
sudo yum install \
        autoconf automake binutils \
        bison flex gcc gcc-c++ gettext \
        libtool make patch pkgconfig \
        redhat-rpm-config rpm-build rpm-sign \
        ctags elfutils indent patchutils 
```

### Build Project

use `cmake` to create Makefile for operators and profiling program:

```bash
# create subdirectory named "build"
mkdir build

# run cmake under [path to root]/build
cd build
cmake ..
```

directory named `bin` and `lib` would be automatically created under root path, then run Makefile to construct final executable.

```bash
# run make under [path to root]/build
make
```

then profiling executables can be obtained under `[path to root]/bin`, operator library can be obtained under `[path to root]/lib`

## Profiling
TODO