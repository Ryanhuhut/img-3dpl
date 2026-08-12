#!/usr/bin/env bash
# Cho build COLMAP xong roi dong goi NGAY - nhung CHI dong goi neu build thanh cong.
# Chay tren may that (khong phai trong container).
set -uo pipefail

BASE=/home/ryanhuhut/bien_dich_cuda
WORK=${BASE}/work
LOG=${WORK}/log-colmap.txt
IMG=colmap-builder:cuda12.2-u2204

echo "=== Cho container colmap-build ket thuc ==="
while docker ps --format '{{.Names}}' | grep -q '^colmap-build$'; do
    sleep 20
done
echo "Container da dung luc $(date '+%H:%M:%S')"
echo "Tien do cuoi cung: $(grep -oE '^\[[0-9]+/[0-9]+\]' "$LOG" | tail -1)"

echo
echo "=== Kiem tra build co thanh cong khong ==="
if [ ! -x "${WORK}/colmap-install/bin/colmap" ]; then
    echo ">>> BUILD THAT BAI: khong thay file colmap-install/bin/colmap"
    echo ">>> 25 dong cuoi cua log:"
    tail -25 "$LOG"
    exit 1
fi
echo "OK: da co ${WORK}/colmap-install/bin/colmap"

echo
echo "=== Dong goi ==="
docker run -i --rm --name colmap-package \
    -v "${WORK}:/work:z" \
    "$IMG" bash -s \
    < "${BASE}/03-package.sh" \
    > "${WORK}/log-package.txt" 2>&1
rc=$?

echo "Ma thoat dong goi: $rc"
if [ $rc -ne 0 ]; then
    echo ">>> DONG GOI THAT BAI - 25 dong cuoi:"
    tail -25 "${WORK}/log-package.txt"
    exit $rc
fi

echo
echo "=== Tra lai quyen so huu cho nguoi dung (container tao file bang root) ==="
docker run --rm -v "${WORK}:/work:z" "$IMG" \
    chown -R "$(id -u):$(id -g)" /work
echo "Xong."

echo
echo "=== KET QUA ==="
ls -lh "${WORK}/dist/"*.tar.gz
