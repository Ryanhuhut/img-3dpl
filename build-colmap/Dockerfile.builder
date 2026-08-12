# Image dung de BIEN DICH COLMAP co CUDA cho Colab.
# Nen: Ubuntu 22.04 (cu hon Colab hoac bang) de nhi phan chay duoc tren ca 22.04 lan 24.04.
# Nhanh "devel" moi co nvcc - trinh bien dich CUDA. Khong can card NVIDIA de dich.
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Chan apt hoi mui gio giua chung lam treo build.
ENV DEBIAN_FRONTEND=noninteractive

# Tat ca ten goi duoi day da duoc kiem chung ton tai trong image nay.
# Co y KHONG cai: qt6-* (da tat GUI), libmkl-full-dev (~3GB, thay bang OpenBLAS).
# Dung ban OpenBLAS "openmp" chu khong phai "pthread" - ban pthread xung dot
# voi OpenMP cua Ceres, gay chay sai ket qua.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libsqlite3-dev \
        libglew-dev \
        libcgal-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libsuitesparse-dev \
        libopenblas-openmp-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Moi thu se build trong /work - thu muc nay se duoc mount tu dia may that,
# nho vay build do dang van chay tiep duoc, va file khong phinh vao image.
WORKDIR /work
