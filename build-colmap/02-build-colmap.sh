#!/usr/bin/env bash
# Build COLMAP 3.13.0 co CUDA, cho GPU T4 (sm75).
# Chay BEN TRONG container colmap-builder, khong chay tren may that.
set -euo pipefail

COLMAP_TAG=3.13.0
CUDA_ARCH=75            # 75 = NVIDIA T4, dung GPU ma Colab bac mien phi cap
JOBS=${JOBS:-3}

# LAN CHAY TRUOC THAT BAI vi CMake tu tai PoseLib tu github.com trong container
# va bi TIMEOUT (mang toi GitHub chap chon). Cach chua: tai san hai goi do tren
# may that (da kiem tra SHA256 khop), roi bao CMake dung ban co san bang hai bien
# FETCHCONTENT_SOURCE_DIR_*. Nho vay container khong con phai ra mang lan nao nua.
echo "=== [0/3] Kiem tra hai goi da tai san ==="
for d in /work/src-poselib /work/src-faiss; do
    if [ ! -f "$d/CMakeLists.txt" ]; then
        echo "THIEU $d - dung lai. Chay lai buoc tai tren may that truoc."
        exit 1
    fi
    echo "OK  $d"
done

echo
echo "=== [1/3] Lay ma nguon COLMAP ${COLMAP_TAG} ==="
cd /work
if [ -d colmap/.git ]; then
    echo "Da co san, bo qua buoc tai."
else
    rm -rf colmap
    git clone --depth 1 --branch "${COLMAP_TAG}" \
        https://github.com/colmap/colmap.git
fi
echo "Tag thuc te dang o: $(git -C /work/colmap describe --tags)"

echo
echo "=== [2/3] Cau hinh ==="
# CMAKE_PREFIX_PATH=/work/deps -> tro toi Ceres 2.2.0 ta tu build o buoc truoc,
#   neu khong CMake se vo tinh bat lay Ceres 2.0 cua he thong va sap.
# GUI_ENABLED=OFF   -> bo Qt6, Colab khong co man hinh.
# OPENGL_ENABLED=OFF-> bo libGL/libglew, bot thu phai cai tren Colab.
# IPO_ENABLED=OFF   -> tat toi uu luc lien ket: ngon RAM, may 11GB de bi giet.
# TESTS_ENABLED=OFF -> khong build bo kiem thu, do mat them thoi gian.
#
# BA CO LIEN QUAN TOI DONG GOI - phai dat NGAY LUC BIEN DICH, sua sau la phai build lai:
# CMAKE_INSTALL_RPATH='$ORIGIN/../lib'
#     Nhung san duong dan tuong doi vao file colmap: "thu vien nam o thu muc lib
#     ben canh thu muc bin cua chinh tao". Nho vay giai nen di dau cung chay.
# CMAKE_BUILD_WITH_INSTALL_RPATH=ON
#     Bat duong dan tren co hieu luc ngay tu luc lien ket.
# -Wl,--disable-new-dtags
#     Ep dung kieu duong dan CU (DT_RPATH) thay vi kieu moi (DT_RUNPATH).
#     Khac biet song con: kieu moi CHI ap dung cho thu vien ma colmap goi truc tiep,
#     khong ap dung cho thu vien ma thu vien khac goi. Vi du libcholmod can libamd,
#     voi kieu moi thi libamd se khong tim thay. Kieu cu ap dung cho tat ca.
cmake -S /work/colmap -B /work/build-colmap -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/work/colmap-install \
    -DCMAKE_PREFIX_PATH=/work/deps \
    -DCUDA_ENABLED=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DGUI_ENABLED=OFF \
    -DOPENGL_ENABLED=OFF \
    -DIPO_ENABLED=OFF \
    -DTESTS_ENABLED=OFF \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags" \
    -DFETCHCONTENT_SOURCE_DIR_POSELIB=/work/src-poselib \
    -DFETCHCONTENT_SOURCE_DIR_FAISS=/work/src-faiss \
    2>&1 | tee /work/cmake-colmap.log

echo
echo "=== KIEM TRA CAU HINH TRUOC KHI TON 2 TIENG BIEN DICH ==="
echo "--- CUDA co that su duoc bat khong? ---"
grep -iE "cuda" /work/cmake-colmap.log | head -20 || true
echo "--- Ceres tim thay o dau? (phai la /work/deps) ---"
grep -iE "ceres" /work/cmake-colmap.log | head -10 || true

echo
echo "=== [3/3] Bien dich (dung ${JOBS} luong) ==="
echo "Bat dau luc: $(date '+%H:%M:%S')"
ninja -C /work/build-colmap -j "${JOBS}"
ninja -C /work/build-colmap install
echo "Ket thuc luc: $(date '+%H:%M:%S')"

echo
echo "=== XONG. Kiem chung: ==="
/work/colmap-install/bin/colmap -h 2>&1 | head -5
