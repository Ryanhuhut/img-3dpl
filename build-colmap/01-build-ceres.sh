#!/usr/bin/env bash
# Build Ceres Solver 2.2.0 tu nguon.
# VI SAO: COLMAP 3.13.0 dung ceres::Manifold - API chi co tu Ceres 2.1 tro len.
# Ubuntu 22.04 chi co Ceres 2.0.0, nen apt khong dung duoc.
# Build kieu TINH (BUILD_SHARED_LIBS=OFF) de Ceres nhung thang vao file colmap,
# nho vay Colab khong can cai Ceres.
set -euo pipefail

CERES_TAG=2.2.0
JOBS=${JOBS:-3}   # 3 chu khong phai 4: chua 1 luong cho may khoi dung hinh

echo "=== [1/3] Lay ma nguon Ceres ${CERES_TAG} ==="
cd /work
if [ -d ceres-solver/.git ]; then
    echo "Da co san, bo qua buoc tai."
else
    rm -rf ceres-solver
    git clone --depth 1 --branch "${CERES_TAG}" \
        https://github.com/ceres-solver/ceres-solver.git
fi
echo "Tag thuc te dang o: $(git -C /work/ceres-solver describe --tags 2>/dev/null || echo '?')"

echo
echo "=== [2/3] Cau hinh ==="
# USE_CUDA=OFF: Ceres khong can CUDA. Nut that cua ta la buoc ghep anh
# (dung nhan CUDA rieng cua COLMAP), khong phai Ceres. Tat di cho nhanh
# va khoi phu thuoc cuSOLVER luc chay.
# POSITION_INDEPENDENT_CODE=ON: bat buoc khi nhung thu vien tinh vao COLMAP.
cmake -S /work/ceres-solver -B /work/build-ceres -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/work/deps \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DUSE_CUDA=OFF \
    -DMINIGLOG=OFF \
    -DGFLAGS=ON \
    -DSUITESPARSE=ON \
    -DLAPACK=ON

echo
echo "=== [3/3] Bien dich va cai vao /work/deps (dung ${JOBS} luong) ==="
ninja -C /work/build-ceres -j "${JOBS}" install

echo
echo "=== XONG. Kiem chung: ==="
ls -la /work/deps/lib/libceres.a 2>&1
grep -r "define CERES_VERSION" /work/deps/include/ceres/version.h 2>/dev/null || true
