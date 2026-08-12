#!/usr/bin/env bash
# Dong goi COLMAP da build thanh mot file .tar.gz tu chua.
# Chay BEN TRONG container colmap-builder.
set -euo pipefail

PREFIX=/work/colmap-install
STAGE=/work/stage/colmap-cuda
OUT=/work/dist

COLMAP_TAG=3.13.0
CUDA_VER=12.2
CUDA_ARCH=75
TARNAME="colmap-${COLMAP_TAG}-cuda${CUDA_VER}-ubuntu2204-sm${CUDA_ARCH}.tar.gz"

# ---------------------------------------------------------------------------
# Danh sach KHONG nhet vao goi. Day la nen he dieu hanh + trinh dieu khien card.
# libcuda.so.1 la trinh dieu khien NVIDIA - BAT BUOC lay tu may co GPU that,
# nhet ban cua may khong co card vao la hong chac chan.
#
# GHI CHU: libgomp DA TUNG nam trong danh sach nay va do la SAI. Khi thu tren
# container ubuntu:22.04 tran, colmap sap voi loi:
#   "error while loading shared libraries: libgomp.so.1"
# libgomp la thu vien chay song song da luong (OpenMP), Ubuntu tran khong co san.
# No khong dinh gi toi phan cung nen nhet vao goi la an toan.
# ---------------------------------------------------------------------------
KHONG_NHET='^(linux-vdso|ld-linux-x86-64|libc|libm|libdl|libpthread|librt|libresolv|libstdc\+\+|libgcc_s|libcuda|libnvidia-.*)\.so'

echo "=== [1/5] Kiem tra da build xong chua ==="
if [ ! -x "${PREFIX}/bin/colmap" ]; then
    echo "KHONG THAY ${PREFIX}/bin/colmap - chua build xong. Dung lai."
    exit 1
fi
echo "OK: $(ls -lh ${PREFIX}/bin/colmap | awk '{print $5}')"

echo
echo "=== [2/5] Dung thu muc goi ==="
rm -rf "${STAGE}" "${OUT}"
mkdir -p "${STAGE}" "${OUT}"
# Dung "/." chu khong dung "/*": kieu "/*" se lam thu muc lib cua COLMAP
# chui vao trong lib da tao san thanh lib/lib. Kieu "/." tron dung cach.
cp -a "${PREFIX}/." "${STAGE}/"
mkdir -p "${STAGE}/lib"

# Bo do thua: file .a la thu vien tinh, include/ la header - ca hai chi can khi
# BIEN DICH phan mem khac dua tren COLMAP, khong can khi CHAY colmap.
# Bo di tiet kiem ~58MB, tuc moi phien Colab tai nhanh hon dang ke.
echo "--- Bo do thua truoc khi nen ---"
echo "Truoc: $(du -sh ${STAGE} | cut -f1)"
rm -f  "${STAGE}"/lib/*.a
rm -rf "${STAGE}/include"
echo "Sau  : $(du -sh ${STAGE} | cut -f1)"

echo
echo "=== [3/5] Do thu vien bang ldd va phan loai ==="
: > /work/lib-da-nhet.txt
: > /work/lib-can-tu-he-thong.txt

ldd "${STAGE}/bin/colmap" | while read -r dong; do
    ten=$(echo "$dong"  | awk '{print $1}')
    duong=$(echo "$dong" | awk '{print $3}')
    [ -z "$ten" ] && continue

    if echo "$ten" | grep -qE "${KHONG_NHET}"; then
        echo "$ten" >> /work/lib-can-tu-he-thong.txt
        continue
    fi
    if [ -z "$duong" ] || [ ! -f "$duong" ]; then
        continue
    fi
    # cp -L: lay file that chu khong lay lien ket tuong trung,
    # nhung giu nguyen TEN ma chuong trinh di tim (SONAME).
    cp -L "$duong" "${STAGE}/lib/${ten}"
    echo "$ten" >> /work/lib-da-nhet.txt
done

echo "--- Da nhet vao goi ($(wc -l < /work/lib-da-nhet.txt) thu vien): ---"
sort /work/lib-da-nhet.txt
echo
echo "--- Colab phai tu co ($(wc -l < /work/lib-can-tu-he-thong.txt) thu vien): ---"
sort /work/lib-can-tu-he-thong.txt

echo
echo "=== [4/5] Ghi BUILD_INFO.txt ==="
{
    echo "COLMAP co CUDA - goi dung lai cho Google Colab"
    echo "=============================================="
    echo
    echo "Ngay build          : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "COLMAP tag          : ${COLMAP_TAG}"
    echo "COLMAP commit       : $(git -C /work/colmap rev-parse HEAD 2>/dev/null || echo '?')"
    echo "CUDA toolkit        : $(nvcc --version | grep release | sed 's/^ *//')"
    echo "Kien truc GPU bat   : sm_${CUDA_ARCH} (NVIDIA T4)"
    echo "He dieu hanh build  : $(grep PRETTY_NAME /etc/os-release | cut -d'\"' -f2)"
    echo "Trinh bien dich     : $(g++ --version | head -1)"
    echo
    echo "--- Thu vien phu thuoc ---"
    echo "Ceres Solver        : 2.2.0 (tu build tu nguon, LIEN KET TINH)"
    echo "                      Ly do: COLMAP 3.13 dung ceres::Manifold,"
    echo "                      chi co tu Ceres 2.1+. Ubuntu 22.04 chi co 2.0.0."
    echo "Boost               : $(dpkg -s libboost-program-options-dev 2>/dev/null | grep ^Version | cut -d' ' -f2)"
    echo "Eigen               : $(dpkg -s libeigen3-dev 2>/dev/null | grep ^Version | cut -d' ' -f2)"
    echo "glog                : $(dpkg -s libgoogle-glog-dev 2>/dev/null | grep ^Version | cut -d' ' -f2)"
    echo "FreeImage           : $(dpkg -s libfreeimage-dev 2>/dev/null | grep ^Version | cut -d' ' -f2)"
    echo "SuiteSparse         : $(dpkg -s libsuitesparse-dev 2>/dev/null | grep ^Version | cut -d' ' -f2)"
    echo "OpenBLAS            : $(dpkg -s libopenblas-openmp-dev 2>/dev/null | grep ^Version | cut -d' ' -f2)"
    echo "PoseLib             : f119951fca625133112acde48daffa5f20eba451"
    echo "faiss               : 36b77353dc435383e0c23a709e7997a29d049041 (ahojnnes/faiss)"
    echo
    echo "--- Tuy chon CMake ---"
    echo "CUDA_ENABLED=ON  CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
    echo "GUI_ENABLED=OFF     (bo Qt6 - Colab khong co man hinh)"
    echo "OPENGL_ENABLED=OFF  (bo libGL/libglew - bot phu thuoc)"
    echo "IPO_ENABLED=OFF     (toi uu luc lien ket ngon RAM, may build chi co 11GB)"
    echo "TESTS_ENABLED=OFF"
    echo "RPATH=\$ORIGIN/../lib voi -Wl,--disable-new-dtags"
    echo "  -> file colmap tu tim thu vien trong thu muc lib ben canh,"
    echo "     ap dung cho ca thu vien goi long nhau."
    echo
    echo "--- Thu vien DA NHET SAN trong goi (thu muc lib/) ---"
    sort /work/lib-da-nhet.txt | sed 's/^/  /'
    echo
    echo "--- Thu vien Colab PHAI TU CO (nen he dieu hanh + driver GPU) ---"
    sort /work/lib-can-tu-he-thong.txt | sed 's/^/  /'
    echo
    echo "--- Cach dung tren Colab ---"
    echo "  tar -xzf ${TARNAME} -C /opt"
    echo "  export PATH=/opt/colmap-cuda/bin:\$PATH"
    echo "  colmap -h    # phai thay chu 'with CUDA'"
    echo
    echo "--- Build lai the nao ---"
    echo "Xem README.md kem theo Release. Toan bo quy trinh nam trong"
    echo "Dockerfile.builder + 01-build-ceres.sh + 02-build-colmap.sh + 03-package.sh"
} > "${STAGE}/BUILD_INFO.txt"

cat "${STAGE}/BUILD_INFO.txt"

echo
echo "=== [5/5] Nen goi ==="
# --owner=0 --group=0: dat quyen so huu ve root trong file nen, neu khong
# se ghi so hieu nguoi dung 1000 - so nay tren Colab co the thuoc ve ai do khac.
tar -czf "${OUT}/${TARNAME}" \
    -C /work/stage \
    --owner=0 --group=0 \
    colmap-cuda

echo
echo "=== XONG ==="
ls -lh "${OUT}/${TARNAME}"
echo "--- BUILD_INFO.txt co trong goi khong? ---"
tar tzf "${OUT}/${TARNAME}" | grep BUILD_INFO
