#!/usr/bin/env python3
"""
Quét 3D — dựng vị trí camera từ file database đã ghép sẵn trên Google Colab.

Cách làm việc:
  1. Trên Colab (có GPU T4): tách đặc trưng + ghép ảnh  →  ra file database.db
  2. Tải database.db về máy
  3. Mở app này: thả thư mục ảnh và file database.db vào, bấm Bắt đầu

Vì sao chia đôi như vậy: ghép ảnh cần GPU, mà máy này không có card NVIDIA.
Còn dựng vị trí camera thì GPU không giúp được gì — nó chạy trên CPU, mà máy
này có 4 luồng trong khi Colab bậc miễn phí chỉ có 2.

Ảnh gốc và file database gốc KHÔNG BAO GIỜ bị ghi đè — cả hai được gắn ở chế
độ chỉ đọc, database được chép sang thư mục kết quả rồi mới dùng.
"""

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, GLib, Gio, Gdk

import concurrent.futures as cf
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
GOI_COLMAP = Path.home() / ".local/share/quet3d/colmap.tar.gz"
IMAGE_DOCKER = "ubuntu:22.04"
TEN_CONTAINER = "quet3d-dang-chay"

DUOI_ANH = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Những hệ thống tệp không giữ được nhãn SELinux riêng cho từng file.
#
# Ổ cắm ngoài và phân vùng dùng chung với Windows hầu hết nằm trong danh sách
# này. Kernel gán cho cả ổ đúng một nhãn chung (exFAT là "dosfs_t") và không
# cách nào đổi được. Docker thì đòi nhãn "container_file_t" mới cho container
# đụng vào, nên mount kiểu thường sẽ bị chặn thẳng: "Permission denied" — dù
# quyền đọc ghi thông thường của thư mục hoàn toàn bình thường.
#
# Gặp ổ như vậy thì phải tắt kiểm soát SELinux cho riêng container này, xem
# _dung_lenh_docker.
HE_TEP_KHONG_NHAN = {
    "exfat", "vfat", "msdos", "ntfs", "ntfs3", "fuseblk", "fuse",
    "iso9660", "udf", "hfs", "hfsplus", "cifs", "smb3", "nfs", "nfs4",
}


def loai_he_thong_tep(duong: Path) -> str:
    """
    Cho biết đường dẫn này nằm trên hệ thống tệp loại gì (ext4, btrfs, exfat…).

    Đọc thẳng /proc/self/mountinfo rồi lấy điểm gắn dài nhất khớp với đường dẫn
    — điểm gắn dài nhất chính là cái đang thực sự chứa file, vì các ổ gắn lồng
    nhau đều là con của "/".
    """
    try:
        duong = duong.resolve()
    except OSError:
        return ""
    dai_nhat, loai = -1, ""
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as f:
            for hang in f:
                truoc, _, sau = hang.partition(" - ")
                cot = truoc.split()
                if len(cot) < 5 or not sau:
                    continue
                diem_gan = Path(cot[4])
                if duong == diem_gan or diem_gan in duong.parents:
                    if len(str(diem_gan)) > dai_nhat:
                        dai_nhat, loai = len(str(diem_gan)), sau.split()[0]
    except OSError:
        return ""
    return loai

# Mã chặng, tên hiển thị, tỉ trọng thời gian.
# Tỉ trọng lấy từ số đo thật trên máy Acer i3 với 320 ảnh:
# dựng camera 105 phút, các chặng còn lại chưa tới 1 phút mỗi chặng.
CHANG = [
    ("chuan_bi", "Chuẩn bị",             2),
    ("dung_cam", "Dựng vị trí camera", 105),
    ("kiem_tra", "Kiểm tra kết quả",     1),
    ("nan_meo",  "Nắn méo ảnh",          2),
    ("nen_zip",  "Nén thành file .zip",  4),
]

# Những thứ được cho vào file .zip mang đi train. Chỉ hai thư mục này, đúng
# cấu trúc Gaussian Splatting đòi. Cố tình bỏ "distorted/" ở ngoài: trong đó
# có bản chép của database.db, nặng hàng GB mà lúc train không đụng tới —
# nhét vào chỉ tổ ngồi đợi tải lên lâu gấp mấy lần.
THU_MUC_MANG_DI = ("images", "sparse")

# Thu nhỏ ảnh trước khi mang lên Colab.
#
# Cỡ ảnh chọn ở đây đi xuyên suốt cả ba chặng sau: database chặng 1 ghi thông
# số camera của đúng cỡ này, image_undistorter chặng 2 xuất ra đúng cỡ này, và
# chặng 3 train ở đúng cỡ này. Muốn train ở 3200px thì phải chọn 3200 NGAY TỪ
# ĐÂY — thêm cờ ở chặng 3 là vô ích, độ phân giải đã bị khoá từ trước rồi.
#
# Vì sao cỡ ảnh KHÔNG phải một hằng số: mỗi loại cảnh có một cỡ đúng riêng, và
# một con số cứng thì loại nào cũng sai một nửa. Bảng dưới đây là bản sao của
# PRESETS[...]["pipeline"]["resize_px"] trong notebook chặng 3 — sửa một bên
# thì phải sửa bên kia, không thì ảnh chụp một cỡ mà train một cỡ khác.
#
#   FLAT_OBJECT     3200  chữ cao 20px trên ảnh 3200px hạ về 1600px chỉ còn
#                         10px, sát ngưỡng Nyquist, qua JPEG nữa là mất hẳn.
#   COMPLEX_OBJECT  2400  nhiều gờ cạnh nhưng ít chữ — gờ cạnh sống sót qua
#                         phép thu nhỏ tốt hơn nét chữ nhiều.
#   ENTIRE_ROOM     1600  phòng cần phủ rộng chứ không cần đọc chữ, mà phòng
#                         thì đi kèm 300-400 tấm nên RAM chặng 3 không kham
#                         nổi cỡ lớn hơn.
#
# 3200px là trần có ích chứ không phải trần tuỳ ý: COLMAP tự hạ ảnh xuống
# max_image_size (mặc định 3200) để dò đặc trưng rồi mới nhân toạ độ keypoint
# trở lại cỡ gốc. Nạp ảnh to hơn 3200px là trả tiền tải lên mà không nhận thêm
# đặc trưng nào.
#
# Đổi lại: ảnh 3200px nặng gấp bốn, chặng 3 train chậm khoảng ba lần, và RAM
# của Colab free chỉ chứa nổi chừng 450 tấm ở cỡ đó. Bù lại thì nên chụp ít
# ảnh hơn — 160 ảnh 3200px về đích nhanh hơn 320 ảnh 1600px mà lại nét hơn,
# vì số cặp ảnh phải ghép ở chặng 1 giảm bốn lần.
PRESET_ANH = {
    "FLAT_OBJECT":    (3200, "Vật thể có chữ nhỏ cần đọc được — 3200 px"),
    "COMPLEX_OBJECT": (2400, "Vật thể nhiều gờ cạnh, ít chữ — 2400 px"),
    "ENTIRE_ROOM":    (1600, "Cả căn phòng — 1600 px"),
}

# Thứ tự hiện trong ô chọn. Cái đầu tiên là mặc định, và mặc định bây giờ là
# FLAT_OBJECT chứ không còn là 1600px: quét vật thể là việc app này làm nhiều
# nhất, mà chọn thiếu độ phân giải thì không có đường sửa ở chặng sau.
THU_TU_PRESET = ["FLAT_OBJECT", "COMPLEX_OBJECT", "ENTIRE_ROOM"]

# Số ảnh nên giữ lại, theo preset. Lấy đúng cận trên của
# PRESETS[...]["pipeline"]["so_anh"] ở notebook chặng 3.
#
# Vì sao phải bớt ảnh, chứ không phải "càng nhiều càng tốt":
#
#   RAM chặng 3.  320 ảnh ở 3200px ăn 7,37 GB RAM (đã tính uint8 và đã dẹp
#                 alpha_mask), mà trần của Colab free là 10,5 GB — biên mỏng
#                 tới mức chỉ cần quên một bản vá là chết giữa buổi train.
#                 180 ảnh còn 4,15 GB, thở được.
#   Chặng 2.      Mapper tăng siêu tuyến tính theo số ảnh: trên con i3 hai
#                 nhân, 320 ảnh mất 105 phút, 180 ảnh còn chừng 35.
#   Chặng 1.      Ghép exhaustive là n(n-1)/2 cặp: 51.040 xuống 16.110.
#
# Và cái giá gần như bằng không: 180 tấm quanh một vật thể là mỗi tấm cách nhau
# 2 độ, trong khi dựng hình chỉ cần 5-10 độ. Phòng thì khác — nó cần phủ rộng
# nên không giảm, xem ENTIRE_ROOM.
SO_ANH_MUC_TIEU = {
    "FLAT_OBJECT": 180,
    "COMPLEX_OBJECT": 220,
    "ENTIRE_ROOM": None,        # phòng cần phủ rộng, bớt tấm nào là hổng chỗ đó
}

# Cạnh của miếng cắt giữa ảnh dùng để chấm điểm độ nét.
#
# Cắt chứ không thu nhỏ, và đây là điểm mấu chốt: thu nhỏ ảnh xuống rồi mới đo
# thì chính những tần số cao phân biệt nét với mờ bị phép thu nhỏ xoá mất, đo
# xong chỉ còn nhiễu. Cắt giữ nguyên tần số gốc, mà lại chỉ phải chập trên 1,4
# triệu điểm ảnh thay vì 12 triệu.
#
# Lấy chính giữa vì quét vòng quanh thì vật thể nằm giữa khung.
CANH_CAT_DO_NET = 1200

# Chất lượng 93 là mức mắt thường không thấy khác mà tệp nhẹ đi mấy lần.
CHAT_LUONG = 93

# Nén .zip thì để nguyên không ép, y như "zip -0" trong script cũ: ảnh JPEG đã
# nén sẵn trong ruột rồi. .tar.gz buộc phải gzip nên để mức 1 cho nhanh.
KIEU_NEN = [
    (".zip",    "Tệp .zip  (mặc định)"),
    (".tar.gz", "Tệp .tar.gz"),
]
TEN_CHANG = {ma: ten for ma, ten, _ in CHANG}
TRONG_SO = {ma: ts for ma, _, ts in CHANG}
TONG_TRONG_SO = sum(TRONG_SO.values())

# Số phút mà 320 ảnh mất ở chặng dựng camera, đo thật trên máy Acer i3-1005G1.
# Dùng làm mốc để ước lượng cho số ảnh khác.
MOC_PHUT = 105
MOC_ANH = 320

# Kịch bản chạy bên trong container.
# Mỗi chặng in ra một dòng ##CHANG:<mã> để giao diện biết đang ở đâu.
# stdbuf -oL ép COLMAP in ra từng dòng ngay lập tức thay vì gom lại,
# nhờ vậy thanh tiến độ nhúc nhích theo thời gian thực.
KICH_BAN = r"""
set -e
echo "##CHANG:chuan_bi"
tar -xzf /pkg/colmap.tar.gz -C /opt
export PATH=/opt/colmap-cuda/bin:$PATH
mkdir -p /out/distorted/sparse

# Chép database sang thư mục kết quả rồi mới dùng. COLMAP có thể ghi vào file
# này lúc chạy, nên tuyệt đối không đụng vào bản gốc người dùng tải từ Colab về.
cp /db/database.db /out/distorted/database.db

echo "##CHANG:dung_cam"
stdbuf -oL -eL colmap mapper \
    --database_path /out/distorted/database.db \
    --image_path /in \
    --output_path /out/distorted/sparse

if [ ! -d /out/distorted/sparse/0 ]; then
    echo "##LOI:COLMAP không dựng được mô hình nào. Ảnh có thể thiếu độ chồng lấn."
    exit 1
fi

echo "##CHANG:kiem_tra"
stdbuf -oL -eL colmap model_analyzer --path /out/distorted/sparse/0

echo "##CHANG:nan_meo"
stdbuf -oL -eL colmap image_undistorter \
    --image_path /in \
    --input_path /out/distorted/sparse/0 \
    --output_path /out \
    --output_type COLMAP
mkdir -p /out/sparse/0
mv /out/sparse/*.bin /out/sparse/0/ 2>/dev/null || true

# Trả quyền sở hữu về cho người dùng: container chạy bằng root nên file nó tạo
# ra thuộc về root, không trả lại thì người dùng không xoá hay sửa được.
#
# Đoạn "|| true" là BẮT BUỘC, không phải cho có: khi người dùng thả một thư mục
# dự án đã có sẵn thư mục con "input", thì thư mục ảnh nằm BÊN TRONG thư mục kết
# quả và đang được gắn ở chế độ chỉ đọc. Lệnh chown sẽ báo lỗi đúng ở chỗ đó.
# Bỏ qua lỗi ấy, vì mọi file ta thực sự tạo ra vẫn được đổi chủ bình thường.
chown -R ${HOST_UID}:${HOST_GID} /out 2>/dev/null || true
echo "##XONG"
"""

CSS = b"""
.canh-bao {
    background-color: #c01c28;
    color: #ffffff;
    border-radius: 14px;
    padding: 20px;
}
.canh-bao-to    { font-size: 27pt; font-weight: 900; letter-spacing: 1px; }
.canh-bao-nho   { font-size: 12pt; }
.dong-ho        { font-size: 46pt; font-weight: 200; font-feature-settings: "tnum"; }
.con-lai        { font-size: 13pt; }
.vung-tha {
    border: 2px dashed alpha(currentColor, 0.35);
    border-radius: 18px;
    padding: 38px 24px;
}
.vung-tha-active {
    border-color: @accent_bg_color;
    background-color: alpha(@accent_bg_color, 0.08);
}
.nhat      { opacity: 0.62; }
.nhat-hon  { opacity: 0.4; }
.log textview { font-family: monospace; font-size: 9pt; }
.bieu-do   { font-family: monospace; font-size: 9pt; opacity: 0.8; }
"""


def tieu_cu_35mm(duong: Path):
    """
    Đọc tiêu cự quy đổi 35mm ghi trong EXIF của một tấm JPEG.

    Tự đọc lấy chứ không nhờ ImageMagick, vì "magick identify" phải giải mã cả
    tấm ảnh mới lấy được EXIF — 236 ảnh mất cả phút đồng hồ, còn đọc thẳng thế
    này chỉ động tới vài KB đầu tệp. (Thử "identify -ping" rồi: nhanh thật
    nhưng không ra EXIF.)

    Trả về số nguyên, hoặc None nếu ảnh không ghi tiêu cự.
    """
    import struct

    try:
        with open(duong, "rb") as f:
            if f.read(2) != b"\xff\xd8":            # không phải JPEG
                return None
            kho = None
            while True:
                dau = f.read(2)
                if len(dau) < 2 or dau[0] != 0xFF:
                    return None
                ma = dau[1]
                if ma == 0xDA:                      # tới phần ảnh, hết chỗ có EXIF
                    return None
                dai = struct.unpack(">H", f.read(2))[0]
                than = f.read(dai - 2)
                if ma == 0xE1 and than[:6] == b"Exif\x00\x00":
                    kho = than[6:]
                    break
            if kho is None:
                return None

        # Bên trong khối Exif là một tệp TIFF thu nhỏ: hai chữ đầu cho biết
        # đọc số theo chiều nào, rồi tới chỗ bắt đầu của bảng thẻ đầu tiên.
        chieu = "<" if kho[:2] == b"II" else ">"
        (goc_ifd,) = struct.unpack_from(chieu + "I", kho, 4)

        def doc_bang(vi_tri, tim):
            (so_the,) = struct.unpack_from(chieu + "H", kho, vi_tri)
            for i in range(so_the):
                o = vi_tri + 2 + i * 12
                the, kieu = struct.unpack_from(chieu + "HH", kho, o)
                if the != tim:
                    continue
                # Kiểu 3 là số 2 byte, kiểu 4 là số 4 byte — cả hai đều nằm
                # gọn trong ô giá trị nên đọc thẳng, khỏi đi tìm đâu xa.
                if kieu == 3:
                    return struct.unpack_from(chieu + "H", kho, o + 8)[0]
                if kieu == 4:
                    return struct.unpack_from(chieu + "I", kho, o + 8)[0]
            return None

        goc_exif = doc_bang(goc_ifd, 0x8769)        # bảng thẻ EXIF nằm riêng
        if goc_exif is None:
            return None
        return doc_bang(goc_exif, 0xA405)           # FocalLengthIn35mmFilm
    except (OSError, struct.error, IndexError):
        return None


def dem_anh(thu_muc: Path) -> int:
    """Đếm số file ảnh nằm trực tiếp trong thư mục."""
    try:
        return sum(1 for f in thu_muc.iterdir()
                   if f.is_file() and f.suffix.lower() in DUOI_ANH)
    except OSError:
        return 0


def kich_thuoc_anh(duong: Path):
    """
    Đọc cỡ thật của một tấm ảnh, tính bằng điểm ảnh. Trả về (rộng, cao) hoặc None.

    Đọc thẳng vài chục byte đầu tệp chứ không nhờ ImageMagick: chặng 2 không hề
    cần ImageMagick, mà bắt cả trang phải có nó chỉ để biết cỡ ảnh thì vô lý.
    Đọc kiểu này cũng không phải giải mã tấm ảnh, nên 320 tấm xong trong chớp
    mắt thay vì cả phút.

    Chỉ hiểu JPEG và PNG. Định dạng khác trả về None, và bên gọi phải coi đó là
    "không biết" chứ không phải "khớp rồi".
    """
    import struct

    try:
        with open(duong, "rb") as f:
            dau = f.read(2)

            # PNG: cỡ ảnh nằm trong khối IHDR, ngay sau 8 byte chữ ký + 8 byte
            # độ dài và tên khối.
            if dau == b"\x89P":
                f.seek(0)
                if f.read(8) != b"\x89PNG\r\n\x1a\n":
                    return None
                f.seek(16)
                than = f.read(8)
                if len(than) < 8:
                    return None
                return struct.unpack(">II", than)

            if dau != b"\xff\xd8":                  # không phải JPEG
                return None

            # JPEG: đi lần lượt qua các khối cho tới khối SOF (khung ảnh).
            # Cỡ ảnh nằm trong SOFn, và chỉ trong SOFn.
            while True:
                b = f.read(1)
                if not b:
                    return None
                if b[0] != 0xFF:                    # lạc nhịp, không đọc tiếp được
                    return None
                ma = f.read(1)
                while ma and ma[0] == 0xFF:         # chuỗi FF đệm, bỏ qua
                    ma = f.read(1)
                if not ma:
                    return None
                ma = ma[0]
                if ma in (0xD8, 0x01) or 0xD0 <= ma <= 0xD7:
                    continue                        # khối không có phần thân
                if ma == 0xDA:                      # tới phần ảnh nén, hết chỗ tìm
                    return None
                dai_byte = f.read(2)
                if len(dai_byte) < 2:
                    return None
                (dai,) = struct.unpack(">H", dai_byte)
                # SOF0-SOF15, trừ ba mã dùng cho việc khác: DHT, JPG, DAC.
                if 0xC0 <= ma <= 0xCF and ma not in (0xC4, 0xC8, 0xCC):
                    than = f.read(5)
                    if len(than) < 5:
                        return None
                    cao, rong = struct.unpack(">HH", than[1:5])
                    return rong, cao
                f.seek(dai - 2, os.SEEK_CUR)
    except (OSError, struct.error):
        return None


def co_anh_trong_thu_muc(thu_muc: Path):
    """
    Những cỡ ảnh có mặt trong một thư mục.

    Trả về (tập hợp (rộng, cao), số tấm đọc được cỡ, số tấm không đọc được).
    Đọc HẾT chứ không lấy mẫu: lấy mẫu thì đúng tấm lạc loài lại là tấm bị bỏ
    qua, mà tấm lạc loài chính là thứ ta đi tìm.
    """
    co, doc_duoc, chiu = set(), 0, 0
    try:
        ds = sorted(f for f in thu_muc.iterdir()
                    if f.is_file() and f.suffix.lower() in DUOI_ANH)
    except OSError:
        return co, 0, 0
    for f in ds:
        kt = kich_thuoc_anh(f)
        if kt is None:
            chiu += 1
        else:
            co.add(kt)
            doc_duoc += 1
    return co, doc_duoc, chiu


def doc_database(duong: Path):
    """
    Đọc thử file database của COLMAP.

    Trả về (số ảnh, số cặp đã ghép, tập hợp cỡ ảnh ghi trong bảng cameras),
    hoặc None nếu file không phải database COLMAP.

    Bảng cameras mới là chỗ đáng đọc nhất, dù trước giờ không ai đọc: nó ghi
    thông số nội tại tính theo ĐIỂM ẢNH của bộ ảnh đã dùng lúc ghép. Đưa cho
    chặng 2 một bộ ảnh cỡ khác thì image_undistorter vẫn chạy êm ru và vẫn xuất
    ra một model — chỉ có điều model đó sai. Không có gì báo lỗi cả.
    """
    try:
        with sqlite3.connect(f"file:{duong}?mode=ro", uri=True) as d:
            c = d.cursor()
            so_anh = c.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            so_cap = c.execute(
                "SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
            co = {(int(r), int(cao)) for r, cao
                  in c.execute("SELECT DISTINCT width, height FROM cameras")}
            return so_anh, so_cap, co
    except sqlite3.Error:
        return None


def ta_co_anh(co) -> str:
    """Viết một tập hợp cỡ ảnh thành chuỗi đọc được: "2400×3200"."""
    return ", ".join(f"{r}×{c}" for r, c in sorted(co)) or "không đọc được"


def do_net(duong: Path):
    """
    Chấm điểm độ nét một tấm ảnh. Trả về số thực, hoặc None nếu đo không được.

    Cách đo là phương sai Laplacian, cách kinh điển để bắt ảnh rung tay hoặc
    lạc nét: Laplacian là đạo hàm bậc hai, nó chỉ nảy lên ở chỗ có biên. Ảnh
    nét đầy biên nên phương sai lớn; ảnh mờ đã bị làm nhẵn nên phương sai bé.
    Đo thử trên cùng một tấm: nét 3212, mờ nhẹ (Gauss 1,2px) 37,9, mờ nặng
    (Gauss 3px) 2,3 — chênh nhau hàng chục lần, không sợ lẫn.

    CON SỐ NÀY CHỈ CÓ NGHĨA KHI SO VỚI NHAU trong cùng một bộ ảnh. Nó phụ thuộc
    vào vật thể chụp cái gì: chụp trang sách đầy chữ thì tấm nào cũng điểm cao
    hơn hẳn chụp một quả cầu nhẵn. Đừng lấy ngưỡng của bộ này áp cho bộ khác.
    """
    lenh = [
        "magick", str(duong),
        "-colorspace", "Gray",
        # Cắt miếng giữa ở ĐỘ PHÂN GIẢI GỐC — xem CANH_CAT_DO_NET.
        "-gravity", "center",
        "-crop", f"{CANH_CAT_DO_NET}x{CANH_CAT_DO_NET}+0+0", "+repage",
        # scale='!' bảo ImageMagick tự co giãn nhân chập cho vừa dải giá trị,
        # không thì phần âm của Laplacian bị kẹp về 0 và mất một nửa tín hiệu.
        "-define", "convolve:scale=!",
        "-morphology", "Convolve", "Laplacian:0",
        "-format", "%[fx:standard_deviation]", "info:",
    ]
    try:
        kq = subprocess.run(lenh, capture_output=True, text=True,
                            env={**os.environ, "MAGICK_THREAD_LIMIT": "1"})
    except OSError:
        return None
    if kq.returncode != 0:
        return None
    try:
        # fx trả về độ lệch chuẩn đã chuẩn hoá về [0,1]. Bình phương lên thành
        # phương sai, rồi nhân 1e6 cho ra số người đọc được thay vì 0,0000023.
        return float(kq.stdout.strip()) ** 2 * 1e6
    except ValueError:
        return None


def bieu_do_net(diem, cot: int = 12) -> str:
    """
    Biểu đồ phân bố độ nét, vẽ bằng chữ.

    Mục đích không phải làm đẹp mà để trả lời một câu: cả bộ ảnh mờ đều, hay
    chỉ vài tấm mờ? Hai chuyện đó chữa bằng hai cách khác hẳn nhau — cả bộ mờ
    thì phải đi chụp lại, còn vài tấm mờ thì bỏ mấy tấm đó là xong.
    """
    co = [d for d in diem if d is not None]
    if not co:
        return "Không chấm điểm được tấm nào."
    it, nhieu = min(co), max(co)
    if nhieu - it < 1e-9:                  # cả bộ y hệt nhau, không có gì để vẽ
        return f"Cả {len(co)} tấm cùng một điểm nét ({it:.0f})."

    thung = [0] * cot
    for d in co:
        i = int((d - it) / (nhieu - it) * cot)
        thung[min(i, cot - 1)] += 1
    cao_nhat = max(thung)

    dong = ["Độ nét (phương sai Laplacian) — càng phải càng nét:"]
    for i, n in enumerate(thung):
        canh_duoi = it + (nhieu - it) * i / cot
        vach = "█" * round(n / cao_nhat * 28) if n else ""
        dong.append(f"  {canh_duoi:8.0f} │{vach:<28} {n}")
    sap = sorted(co)
    giua = sap[len(sap) // 2]
    dong.append(f"  thấp nhất {it:.0f} · trung vị {giua:.0f} · cao nhất {nhieu:.0f}")
    if it < giua / 4:
        dong.append("  Đuôi trái dài — có mấy tấm mờ hẳn so với phần còn lại.")
    return "\n".join(dong)


def chon_giu(ds, diem, muc_tieu):
    """
    Chọn muc_tieu tấm trong ds, GIỮ NGUYÊN độ phủ vòng tròn.

    Chia danh sách thành đúng muc_tieu khoảng liền nhau, đều nhau, rồi mỗi
    khoảng giữ lại tấm nét nhất. Danh sách đã sắp theo tên tệp, mà tên tệp máy
    ảnh đánh theo thứ tự bấm máy, nên "khoảng liền nhau" cũng chính là "cung
    liền nhau" trên vòng quét.

    Vì sao KHÔNG được lấy N tấm đầu danh sách: quét vòng quanh vật thể thì nửa
    sau danh sách là nửa sau vòng tròn. Cắt đuôi là mất hẳn một bên vật thể, và
    COLMAP sẽ dựng ra đúng một nửa mô hình.

    Vì sao không lấy cách đều máy móc (tấm 1, 3, 5...): cách đều thì giữ đúng
    độ phủ nhưng gặp tấm mờ vẫn phải lấy. Chia khoảng rồi chọn tấm nét nhất
    trong khoảng thì vừa giữ độ phủ, vừa tự loại tấm rung tay — với muc_tieu
    bằng một nửa số ảnh, nó chính là "mỗi cặp giữ tấm nét hơn".
    """
    n = len(ds)
    if not muc_tieu or muc_tieu >= n:
        return list(ds)
    giu = []
    for i in range(muc_tieu):
        dau, cuoi = i * n // muc_tieu, (i + 1) * n // muc_tieu
        # Tấm không chấm điểm được coi như kém nhất, nhưng vẫn được lấy nếu cả
        # khoảng chẳng tấm nào chấm được — thà giữ độ phủ còn hơn thủng một cung.
        giu.append(max(range(dau, cuoi),
                       key=lambda k: -1.0 if diem[k] is None else diem[k]))
    return [ds[k] for k in giu]


def doc_thoi_gian(giay: float) -> str:
    """Đổi số giây thành dạng 1:23:45 hoặc 23:45."""
    gio, du = divmod(int(max(giay, 0)), 3600)
    phut, gy = divmod(du, 60)
    return f"{gio}:{phut:02d}:{gy:02d}" if gio else f"{phut}:{gy:02d}"


def doc_con_lai(giay: float) -> str:
    """Đổi số giây còn lại thành câu dễ đọc cho người không rành máy tính."""
    giay = max(giay, 0)
    if giay < 90:
        return "còn dưới 2 phút"
    phut = round(giay / 60)
    if phut < 60:
        return f"còn khoảng {phut} phút"
    gio = giay / 3600
    if gio < 2:
        return f"còn khoảng 1 tiếng {round((gio - 1) * 60)} phút"
    return f"còn khoảng {gio:.1f} tiếng".replace(".", ",")


class TrangNenAnh(Gtk.Box):
    """
    Thu nhỏ ảnh theo cỡ của preset rồi gói lại thành một tệp mang lên Colab.

    Gói PHẲNG: ảnh nằm thẳng ở gốc tệp nén, không có thư mục bọc bên ngoài —
    giải nén ra là thấy ảnh ngay. Khác với "zip -r" trong script cũ, vốn bọc
    thêm một lớp thư mục rồi lên Colab lại phải đi tìm.

    Ảnh gốc không bị đụng tới: ảnh thu nhỏ ghi ra thư mục tạm, gói xong thì
    xoá thư mục tạm đi, chỉ còn lại đúng một tệp nén.
    """

    def __init__(self, cua):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.cua = cua
        self.dang_chay = False
        self.bi_huy = False
        self.dich_tu_dien = ""      # đường dẫn ra do tool tự điền, chưa ai sửa
        # Cỡ ảnh của lần nén đang chạy, chốt theo preset đang chọn.
        self.canh = PRESET_ANH[THU_TU_PRESET[0]][0]
        self.muc_tieu = SO_ANH_MUC_TIEU[THU_TU_PRESET[0]]

        cuon = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER,
                                  vexpand=True)
        hop = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16,
                      margin_top=20, margin_bottom=20,
                      margin_start=24, margin_end=24)
        cuon.set_child(hop)
        self.append(cuon)

        loi_nhac = Gtk.Label(
            label="Thu nhỏ ảnh rồi gói thành một tệp để tải lên Colab cho "
                  "nhanh. Giải nén ra là ảnh nằm luôn ở ngoài, không có thư "
                  "mục bọc. Ảnh gốc giữ nguyên.",
            wrap=True, xalign=0)
        loi_nhac.add_css_class("nhat")
        hop.append(loi_nhac)

        nhom = Adw.PreferencesGroup()
        self.o_nguon = Adw.EntryRow(title="Đường dẫn thư mục ảnh gốc")
        self.o_nguon.add_suffix(self._nut_duyet(self._chon_nguon))
        self.o_nguon.connect("changed", self._khi_doi_nguon)
        nhom.add(self.o_nguon)

        self.o_dich = Adw.EntryRow(title="Đường dẫn tệp nén sẽ tạo ra")
        self.o_dich.add_suffix(self._nut_duyet(self._chon_dich))
        nhom.add(self.o_dich)

        # Chọn loại cảnh ở đây là chốt luôn cỡ ảnh cho cả ba chặng sau — xem
        # PRESET_ANH. Người dùng chọn thứ mình biết ("tôi chụp cái gì"), không
        # phải thứ mình phải tự suy ra ("nên để bao nhiêu pixel").
        self.o_canh = Adw.ComboRow(
            title="Loại cảnh đang quét",
            subtitle="Quyết định cỡ ảnh, và cỡ ảnh thì không sửa lại được ở "
                     "chặng sau",
            model=Gtk.StringList.new(
                [PRESET_ANH[ten][1] for ten in THU_TU_PRESET]))
        self.o_canh.connect("notify::selected", self._khi_doi_canh)
        nhom.add(self.o_canh)

        self.o_kieu = Adw.ComboRow(
            title="Kiểu nén",
            model=Gtk.StringList.new([ten for _, ten in KIEU_NEN]))
        self.o_kieu.connect("notify::selected", self._khi_doi_kieu)
        nhom.add(self.o_kieu)

        # Giảm bớt ảnh. Bật sẵn hay không là theo preset: vật thể thì thừa ảnh,
        # phòng thì thiếu — xem SO_ANH_MUC_TIEU.
        self.o_giam = Adw.SwitchRow(
            title="Giảm bớt số ảnh",
            subtitle="Chia đều vòng quét rồi mỗi khoảng giữ tấm nét nhất — "
                     "không cắt đầu, không cắt đuôi",
            active=SO_ANH_MUC_TIEU[THU_TU_PRESET[0]] is not None)
        self.o_giam.connect("notify::active", self._khi_doi_giam)
        nhom.add(self.o_giam)

        self.o_so_anh = Adw.SpinRow.new_with_range(20, 2000, 10)
        self.o_so_anh.set_title("Giữ lại bao nhiêu ảnh")
        self.o_so_anh.set_subtitle("Nhiều hơn số ảnh đang có thì giữ nguyên tất")
        self.o_so_anh.set_value(SO_ANH_MUC_TIEU[THU_TU_PRESET[0]] or 180)
        nhom.add(self.o_so_anh)

        # Bật sẵn, và nên để yên: chặng dựng camera cần đúng bộ ảnh đã thu nhỏ
        # này chứ không phải ảnh gốc. Tắt đi thì lát nữa phải tự giải nén ra lại.
        self.o_giu = Adw.SwitchRow(
            title="Giữ lại thư mục ảnh đã thu nhỏ",
            subtitle="Bước dựng vị trí camera cần đúng bộ ảnh này",
            active=True)
        nhom.add(self.o_giu)
        hop.append(nhom)

        self.nhan_tt = Gtk.Label(label="", wrap=True, xalign=0, visible=False)
        self.nhan_tt.add_css_class("nhat")
        hop.append(self.nhan_tt)

        # Biểu đồ phân bố độ nét. Phải là chữ đơn cách, không thì các vạch so le.
        self.nhan_bieu_do = Gtk.Label(label="", xalign=0, visible=False,
                                      selectable=True)
        self.nhan_bieu_do.add_css_class("bieu-do")
        hop.append(self.nhan_bieu_do)

        self.thanh = Gtk.ProgressBar(show_text=True, visible=False)
        hop.append(self.thanh)

        hop_nut = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10,
                          halign=Gtk.Align.CENTER, margin_top=4)
        self.nut_chay = Gtk.Button(label="Bắt đầu nén")
        self.nut_chay.add_css_class("suggested-action")
        self.nut_chay.add_css_class("pill")
        self.nut_chay.connect("clicked", self._bam_bat_dau)
        hop_nut.append(self.nut_chay)
        self.nut_huy = Gtk.Button(label="Dừng lại", visible=False)
        self.nut_huy.add_css_class("destructive-action")
        self.nut_huy.add_css_class("pill")
        self.nut_huy.connect("clicked", self._bam_huy)
        hop_nut.append(self.nut_huy)
        hop.append(hop_nut)

        self._khi_doi_giam()

        if shutil.which("magick") is None:
            self.nut_chay.set_sensitive(False)
            self._bao("Máy chưa cài ImageMagick — cần lệnh “magick” để thu nhỏ ảnh")

    def _nut_duyet(self, ham):
        nut = Gtk.Button(icon_name="folder-open-symbolic",
                         valign=Gtk.Align.CENTER, tooltip_text="Duyệt…")
        nut.add_css_class("flat")
        nut.connect("clicked", ham)
        return nut

    def _bao(self, chu: str):
        self.cua._bao(chu)

    # ---------------------------------------------------------- chọn đường dẫn
    def _chon_nguon(self, *_):
        hop = Gtk.FileDialog(title="Chọn thư mục ảnh gốc")
        cu = Path(self.o_nguon.get_text().strip()).expanduser()
        if cu.is_dir():
            hop.set_initial_folder(Gio.File.new_for_path(str(cu)))
        hop.select_folder(self.cua, None, self._nhan_nguon)

    def _nhan_nguon(self, hop, ket_qua):
        try:
            tep = hop.select_folder_finish(ket_qua)
        except GLib.Error:
            return
        if tep:
            self.o_nguon.set_text(tep.get_path())

    def _chon_dich(self, *_):
        hop = Gtk.FileDialog(title="Lưu tệp nén vào đâu")
        cu = Path(self.o_dich.get_text().strip()).expanduser()
        if cu.name:
            hop.set_initial_name(cu.name)
        if cu.parent.is_dir():
            hop.set_initial_folder(Gio.File.new_for_path(str(cu.parent)))
        hop.save(self.cua, None, self._nhan_dich)

    def _nhan_dich(self, hop, ket_qua):
        try:
            tep = hop.save_finish(ket_qua)
        except GLib.Error:
            return
        if tep:
            self.o_dich.set_text(tep.get_path())

    def _duoi_dang_chon(self) -> str:
        return KIEU_NEN[self.o_kieu.get_selected()][0]

    def _preset(self) -> str:
        """Tên preset đang chọn, đúng tên dùng ở notebook chặng 3."""
        return THU_TU_PRESET[self.o_canh.get_selected()]

    def _canh(self) -> int:
        """Cạnh dài đang chọn, tính bằng pixel. Suy ra từ preset, không nhập tay."""
        return PRESET_ANH[self._preset()][0]

    def _khi_doi_nguon(self, *_):
        """Tự điền đường dẫn ra theo tên thư mục ảnh — trừ khi người dùng đã sửa."""
        o_dich = self.o_dich.get_text().strip()
        if o_dich and o_dich != self.dich_tu_dien:
            return
        nguon = Path(self.o_nguon.get_text().strip()).expanduser()
        if not nguon.name:
            return
        self.dich_tu_dien = str(nguon.parent /
                                f"{nguon.name}_{self._canh()}{self._duoi_dang_chon()}")
        self.o_dich.set_text(self.dich_tu_dien)

    def _khi_doi_canh(self, *_):
        """
        Đổi preset thì đổi luôn tên tệp ra, để hai bộ khác cỡ không đè nhau.

        Và đổi luôn số ảnh mục tiêu: phòng thì không giảm, vật thể thì giảm —
        hai chuyện này đi liền nhau, bắt người dùng nhớ chỉnh cả hai là có ngày
        quét phòng mà chỉ còn 180 tấm.
        """
        self._khi_doi_nguon()
        muc_tieu = SO_ANH_MUC_TIEU[self._preset()]
        self.o_giam.set_active(muc_tieu is not None)
        if muc_tieu is not None:
            self.o_so_anh.set_value(muc_tieu)

    def _khi_doi_giam(self, *_):
        self.o_so_anh.set_sensitive(self.o_giam.get_active())

    def _khi_doi_kieu(self, *_):
        """Đổi kiểu nén thì thay luôn phần đuôi của đường dẫn ra."""
        hien = self.o_dich.get_text().strip()
        if not hien:
            return
        for duoi, _ in KIEU_NEN:
            if hien.endswith(duoi):
                hien = hien[:-len(duoi)]
                break
        moi = hien + self._duoi_dang_chon()
        if self.o_dich.get_text().strip() == self.dich_tu_dien:
            self.dich_tu_dien = moi
        self.o_dich.set_text(moi)

    # ------------------------------------------------------------------ chạy
    def _bam_bat_dau(self, *_):
        nguon = Path(self.o_nguon.get_text().strip()).expanduser()
        dich = Path(self.o_dich.get_text().strip()).expanduser()

        if not nguon.is_dir():
            self._bao("Dòng trên chưa trỏ tới một thư mục có thật")
            return
        ds = sorted(f for f in nguon.iterdir()
                    if f.is_file() and f.suffix.lower() in DUOI_ANH)
        if not ds:
            self._bao(f"Không thấy ảnh nào trong “{nguon.name}”")
            return
        if not dich.name or dich.name.startswith("."):
            self._bao("Dòng dưới chưa có tên tệp nén")
            return
        if not dich.parent.is_dir():
            self._bao(f"Không có thư mục “{dich.parent}” để lưu vào")
            return
        if not os.access(dich.parent, os.W_OK):
            self._bao(f"Không có quyền ghi vào “{dich.parent}”")
            return
        if dich.exists():
            self._hoi_ghi_de(nguon, dich, ds)
            return
        self._chay(nguon, dich, ds)

    def _hoi_ghi_de(self, nguon, dich, ds):
        hop = Adw.AlertDialog(
            heading="Đã có tệp này rồi",
            body=f"“{dich.name}” đang tồn tại. Nén tiếp là đè lên bản cũ.")
        hop.add_response("khong", "Thôi")
        hop.add_response("co", "Đè lên")
        hop.set_response_appearance("co", Adw.ResponseAppearance.DESTRUCTIVE)
        hop.set_default_response("khong")
        hop.connect("response",
                    lambda _d, r: self._chay(nguon, dich, ds) if r == "co" else None)
        hop.present(self.cua)

    def _chay(self, nguon, dich, ds):
        self.dang_chay, self.bi_huy = True, False
        # Chốt cỡ ảnh ngay tại đây, khi còn đang ở luồng giao diện. Luồng nén
        # chạy nền không được phép đọc trạng thái widget.
        self.canh = self._canh()
        self.muc_tieu = (int(self.o_so_anh.get_value())
                         if self.o_giam.get_active() else None)
        self.nhan_bieu_do.set_visible(False)
        # Khoá nút quay lại: đang nén dở mà bỏ đi trang khác thì lát nữa quay
        # vào chẳng biết nó chạy tới đâu. Muốn thoát thì bấm "Dừng lại".
        self.cua.nut_quay_lai.set_sensitive(False)
        self.nut_chay.set_visible(False)
        self.nut_huy.set_visible(True)
        self.nut_huy.set_sensitive(True)
        for o in (self.o_nguon, self.o_dich, self.o_kieu, self.o_canh,
                  self.o_giam, self.o_so_anh):
            o.set_sensitive(False)
        self.nhan_tt.set_visible(True)
        self.thanh.set_visible(True)
        self._dat_tien_do(0, len(ds), "đang xem tiêu cự EXIF…")
        threading.Thread(target=self._luong, args=(nguon, dich, ds),
                         daemon=True).start()

    def _bam_huy(self, *_):
        self.bi_huy = True
        self.nut_huy.set_sensitive(False)
        self.nut_huy.set_label("Đang dừng…")

    def _dat_tien_do(self, xong: int, tong: int, chu: str):
        self.thanh.set_fraction(xong / tong if tong else 0)
        self.thanh.set_text(f"{xong} / {tong} ảnh")
        self.nhan_tt.set_label(chu)
        return False

    def _xem_tieu_cu(self, ds):
        """
        Đếm xem cả bộ ảnh có chung một tiêu cự không, giống script cũ.

        Lẫn ảnh chụp bằng nhiều mức zoom khác nhau là COLMAP dựng ra mô hình
        cong vênh, mà tới lúc ấy đã mất mấy tiếng rồi. Biết trước vẫn hơn.
        """
        return {str(t) for f in ds if (t := tieu_cu_35mm(f)) is not None}

    def _thu_nho(self, vao: Path, ra: Path):
        """Thu nhỏ một tấm. Trả về lời báo lỗi, hoặc None nếu êm xuôi."""
        if self.bi_huy:
            return None
        try:
            kq = subprocess.run(
                ["magick", str(vao), "-resize", f"{self.canh}x{self.canh}",
                 "-quality", str(CHAT_LUONG), str(ra)],
                capture_output=True, text=True,
                # Mỗi tấm để ImageMagick chạy một luồng thôi, vì ta đã cho chạy
                # nhiều tấm cùng lúc rồi — không thì 4 nhân giành nhau, chậm hơn.
                env={**os.environ, "MAGICK_THREAD_LIMIT": "1"})
        except OSError as e:
            return f"{vao.name}: {e}"
        if kq.returncode != 0:
            return f"{vao.name}: {kq.stderr.strip().splitlines()[-1:] or ''}"
        return None

    def _don_rac_cu(self, cho: Path):
        """
        Dọn thư mục tạm mà lần chạy trước bỏ lại.

        Nén xong là tự xoá, nên chỉ khi app bị giết ngang (mất điện, tắt máy)
        mới còn sót. Phải quá hai tiếng không ai sờ tới mới dám dọn, phòng
        trường hợp có cửa sổ khác đang nén dở vào cùng chỗ.
        """
        gio = time.time()
        for d in cho.glob(".nen-anh-*"):
            try:
                if d.is_dir() and gio - d.stat().st_mtime > 7200:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass

    def _cham_diem_ca_bo(self, ds):
        """
        Chấm điểm độ nét cả bộ ảnh. Trả về danh sách điểm, cùng thứ tự với ds.

        Chạy trước khi thu nhỏ, và phải như vậy: thu nhỏ xong thì tấm rung tay
        với tấm nét trông na ná nhau, đúng những tần số phân biệt chúng vừa bị
        phép thu nhỏ xoá đi.
        """
        diem = [None] * len(ds)
        xong = 0
        with cf.ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as bom:
            viec = {bom.submit(do_net, f): i for i, f in enumerate(ds)}
            for v in cf.as_completed(viec):
                if self.bi_huy:
                    bom.shutdown(cancel_futures=True)
                    break
                diem[viec[v]] = v.result()
                xong += 1
                GLib.idle_add(self._dat_tien_do, xong, len(ds),
                              "đang chấm điểm độ nét…")
        return diem

    def _hien_bieu_do(self, chu: str):
        self.nhan_bieu_do.set_label(chu)
        self.nhan_bieu_do.set_visible(True)
        return False

    def _luong(self, nguon: Path, dich: Path, ds):
        self._don_rac_cu(dich.parent)
        tam = Path(tempfile.mkdtemp(prefix=".nen-anh-", dir=dich.parent))
        try:
            muc = self._xem_tieu_cu(ds)
            if muc and len(muc) > 1:
                GLib.idle_add(
                    self._bao, f"Ảnh có {len(muc)} mức tiêu cự khác nhau "
                               f"({', '.join(sorted(muc))}) — COLMAP dễ dựng lệch")

            # Chấm điểm độ nét cho MỌI lần chạy, kể cả khi không giảm ảnh:
            # biểu đồ phân bố trả lời câu "cả bộ mờ đều hay chỉ vài tấm mờ",
            # mà đó là câu đáng biết trước khi ngồi chờ ba tiếng ở chặng 2.
            # Nó tốn thêm chừng một phần tư giây mỗi tấm, so với vài phút của
            # phần thu nhỏ ngay dưới thì không đáng kể.
            diem = self._cham_diem_ca_bo(ds)
            if self.bi_huy:
                raise InterruptedError
            GLib.idle_add(self._hien_bieu_do, bieu_do_net(diem))

            bo_di = 0
            if self.muc_tieu and self.muc_tieu < len(ds):
                bo_di = len(ds) - self.muc_tieu
                ds = chon_giu(ds, diem, self.muc_tieu)

            tong = len(ds)
            xong, loi = 0, []
            with cf.ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as bom:
                viec = [bom.submit(self._thu_nho, f, tam / f.name) for f in ds]
                for v in cf.as_completed(viec):
                    if self.bi_huy:
                        bom.shutdown(cancel_futures=True)
                        break
                    if (e := v.result()):
                        loi.append(e)
                    xong += 1
                    GLib.idle_add(self._dat_tien_do, xong, tong,
                                  f"đang thu nhỏ còn {self.canh}px…")
            if self.bi_huy:
                raise InterruptedError

            GLib.idle_add(self._dat_tien_do, tong, tong,
                          f"đang gói vào {dich.name}…")
            self._goi(tam, dich)
        except InterruptedError:
            GLib.idle_add(self._xong, dich, None, "Đã dừng giữa chừng")
            return
        except Exception as e:                                    # noqa: BLE001
            GLib.idle_add(self._xong, dich, None, str(e))
            return
        finally:
            shutil.rmtree(tam, ignore_errors=True)
        GLib.idle_add(self._xong, dich, loi, None, tong, bo_di)

    def _goi(self, tam: Path, dich: Path):
        """Gói phẳng: mỗi ảnh vào tệp nén bằng đúng cái tên của nó, không kèm lối."""
        ds = sorted(f for f in tam.iterdir() if f.is_file())
        cho = dich.with_name(dich.name + ".dang-ghi")
        if dich.name.endswith(".tar.gz"):
            with tarfile.open(cho, "w:gz", compresslevel=1) as t:
                for f in ds:
                    t.add(f, arcname=f.name)
        else:
            with zipfile.ZipFile(cho, "w", zipfile.ZIP_STORED) as z:
                for f in ds:
                    z.write(f, f.name)
        cho.replace(dich)

    def _xong(self, dich: Path, loi, hong: str | None,
              con_lai: int = 0, bo_di: int = 0):
        self.dang_chay = False
        self.cua.nut_quay_lai.set_sensitive(True)
        self.nut_huy.set_visible(False)
        self.nut_huy.set_label("Dừng lại")
        self.nut_chay.set_visible(True)
        for o in (self.o_nguon, self.o_dich, self.o_kieu, self.o_canh,
                  self.o_giam, self.o_so_anh):
            o.set_sensitive(True)
        # Ô số ảnh chỉ mở khi công tắc giảm ảnh đang bật — vòng lặp trên mở
        # tuốt, nên phải trả lại đúng trạng thái ở đây.
        self._khi_doi_giam()
        self.thanh.set_visible(False)

        if hong:
            self.nhan_tt.set_label(f"Không xong: {hong}")
            self._bao(f"Không xong: {hong}")
            return
        cd = dich.stat().st_size / 1e6 if dich.is_file() else 0
        chu = f"Xong: {dich.name} · {cd:.0f} MB"
        if bo_di:
            chu += f" · giữ {con_lai} ảnh, bỏ bớt {bo_di} tấm kém nét hơn"
        if loi:
            chu += f" · {len(loi)} ảnh lỗi bị bỏ qua"
        self.nhan_tt.set_label(chu + f"\n{dich}")
        self._bao(chu)
        return False


class CuaSo(Adw.ApplicationWindow):

    def __init__(self, app):
        super().__init__(application=app, title="Quét 3D")
        self.set_default_size(800, 800)

        self.app = app
        self.thu_muc_goc: Path | None = None    # thư mục người dùng thả vào
        self.thu_muc_anh: Path | None = None    # thư mục thật sự chứa ảnh
        self.thu_muc_dich: Path | None = None   # nơi lưu do người dùng chọn
        self.thu_muc_ra: Path | None = None
        self.file_zip: Path | None = None
        self.file_db: Path | None = None
        self.so_anh = 0
        self.so_anh_db = 0
        self.co_anh_thuc: set = set()       # cỡ ảnh đọc từ chính các tấm ảnh
        self.so_anh_kho_doc = 0            # tấm không đọc nổi cỡ (không phải JPEG/PNG)
        self.co_anh_db: set = set()        # cỡ ảnh ghi trong bảng cameras của .db

        self.dang_chay = False
        self.dang_nen = False
        self.bi_huy = False
        self.tien_trinh: subprocess.Popen | None = None
        self.luc_bat_dau = 0.0
        self.luc_co_tin = 0.0
        self.luc_bat_dau_chang = 0.0
        self.chang_hien_tai: str | None = None
        self.phan_tram_chang = 0.0
        self.trong_so_da_qua = 0
        self.id_dong_ho: int | None = None
        self.khoa_ngu = None

        self.toast = Adw.ToastOverlay()
        self.set_content(self.toast)

        khung = Adw.ToolbarView()
        self.header = Adw.HeaderBar()
        self.tieu_de = Adw.WindowTitle(title="Quét 3D")
        self.header.set_title_widget(self.tieu_de)

        self.nut_nen_anh = Gtk.Button(label="Nén ảnh")
        self.nut_nen_anh.add_css_class("pill")
        self.nut_nen_anh.add_css_class("flat")
        self.nut_nen_anh.set_tooltip_text(
            "Thu nhỏ ảnh rồi gói lại thành một tệp để tải lên Colab")
        self.nut_nen_anh.connect("clicked", lambda *_: self._sang_trang("nen"))
        self.header.pack_start(self.nut_nen_anh)

        # Bấm nhầm vào nút nén thì quay ra ngay, khỏi phải tắt app mở lại.
        self.nut_quay_lai = Gtk.Button(icon_name="go-previous-symbolic",
                                       tooltip_text="Quay lại", visible=False)
        self.nut_quay_lai.add_css_class("flat")
        self.nut_quay_lai.connect("clicked", lambda *_: self._sang_trang("cho"))
        self.header.pack_start(self.nut_quay_lai)

        khung.add_top_bar(self.header)
        self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        khung.set_content(self.stack)
        self.toast.set_child(khung)

        self.stack.add_named(self._trang_cho(), "cho")
        self.trang_nen = TrangNenAnh(self)
        self.stack.add_named(self.trang_nen, "nen")
        self.stack.add_named(self._trang_chay(), "chay")

        # Bấm Escape hay nút "quay lại" trên chuột cũng ra được, như mọi app khác
        phim = Gtk.EventControllerKey()
        phim.connect("key-pressed", self._khi_bam_phim)
        self.add_controller(phim)

        self.connect("close-request", self._khi_dong_cua_so)

    def _sang_trang(self, ten: str):
        """Đổi trang và sửa lại thanh tiêu đề cho khớp."""
        self.stack.set_visible_child_name(ten)
        self.nut_nen_anh.set_visible(ten == "cho")
        self.nut_quay_lai.set_visible(ten == "nen")
        self.tieu_de.set_title("Nén ảnh cho Colab" if ten == "nen"
                               else "Quét 3D")

    def _khi_bam_phim(self, _bo, phim, _ma, _trang_thai):
        if (phim == Gdk.KEY_Escape
                and self.stack.get_visible_child_name() == "nen"
                and self.nut_quay_lai.get_sensitive()):
            self._sang_trang("cho")
            return True
        return False

    # --------------------------------------------------------- trang lúc chờ
    def _trang_cho(self):
        cuon = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER)
        hop = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18,
                      margin_top=24, margin_bottom=24,
                      margin_start=28, margin_end=28,
                      valign=Gtk.Align.CENTER)
        cuon.set_child(hop)

        # Nhắc lại quy trình, để vài tháng sau mở lại vẫn nhớ mình đang ở đâu
        nhac = Adw.Banner(
            title="Ghép ảnh chạy trên Colab trước. App này làm khâu sau đó.",
            revealed=True)
        hop.append(nhac)

        self.vung_tha = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.vung_tha.add_css_class("vung-tha")

        bieu_tuong = Gtk.Image.new_from_icon_name("folder-download-symbolic")
        bieu_tuong.set_pixel_size(60)
        bieu_tuong.add_css_class("nhat")
        self.vung_tha.append(bieu_tuong)

        self.nhan_tha = Gtk.Label(label="Thả thư mục ảnh và file .db vào đây")
        self.nhan_tha.add_css_class("title-2")
        self.nhan_tha.set_wrap(True)
        self.nhan_tha.set_justify(Gtk.Justification.CENTER)
        self.vung_tha.append(self.nhan_tha)

        self.nhan_phu = Gtk.Label(label="thả từng thứ một cũng được")
        self.nhan_phu.add_css_class("nhat")
        self.vung_tha.append(self.nhan_phu)

        hop_nut = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10,
                          halign=Gtk.Align.CENTER, margin_top=8)
        nut_anh = Gtk.Button(label="Chọn thư mục ảnh…")
        nut_anh.add_css_class("pill")
        nut_anh.connect("clicked", self._mo_chon_anh)
        hop_nut.append(nut_anh)
        nut_db = Gtk.Button(label="Chọn file .db…")
        nut_db.add_css_class("pill")
        nut_db.connect("clicked", self._mo_chon_db)
        hop_nut.append(nut_db)
        nut_dich = Gtk.Button(label="Chọn nơi lưu…")
        nut_dich.add_css_class("pill")
        nut_dich.connect("clicked", self._mo_chon_dich)
        hop_nut.append(nut_dich)
        self.vung_tha.append(hop_nut)

        # Kéo thả: GTK4 trả về danh sách file qua Gdk.FileList
        tha = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        tha.connect("drop", self._khi_tha)
        tha.connect("enter", self._khi_ro_vao)
        tha.connect("leave", self._khi_ro_ra)
        self.vung_tha.add_controller(tha)
        hop.append(self.vung_tha)

        self.nhom_tt = Adw.PreferencesGroup(title="Đầu vào")
        self.hang_anh = Adw.ActionRow(title="Thư mục ảnh", subtitle="chưa chọn")
        self.hang_anh.add_prefix(
            Gtk.Image.new_from_icon_name("image-x-generic-symbolic"))
        self.hang_db = Adw.ActionRow(title="File database từ Colab",
                                     subtitle="chưa chọn")
        self.hang_db.add_prefix(
            Gtk.Image.new_from_icon_name("drive-harddisk-symbolic"))
        # Bấm thẳng vào hàng này cũng đổi được nơi lưu, khỏi phải mò lên nút.
        self.hang_ra = Adw.ActionRow(title="File mang đi train", subtitle="—")
        self.hang_ra.add_prefix(Gtk.Image.new_from_icon_name("folder-symbolic"))
        self.hang_ra.add_suffix(
            Gtk.Image.new_from_icon_name("document-edit-symbolic"))
        self.hang_ra.set_activatable(True)
        self.hang_ra.connect("activated", self._mo_chon_dich)
        for h in (self.hang_anh, self.hang_db, self.hang_ra):
            self.nhom_tt.add(h)
        hop.append(self.nhom_tt)

        self.nhom_ul = Adw.PreferencesGroup()
        self.hang_uoc_luong = Adw.ActionRow(title="Chưa đủ đầu vào")
        self.hang_uoc_luong.add_prefix(
            Gtk.Image.new_from_icon_name("preferences-system-time-symbolic"))
        self.nhom_ul.add(self.hang_uoc_luong)
        hop.append(self.nhom_ul)

        self.nut_bat_dau = Gtk.Button(label="Bắt đầu dựng mô hình",
                                      halign=Gtk.Align.CENTER, margin_top=6)
        self.nut_bat_dau.add_css_class("suggested-action")
        self.nut_bat_dau.add_css_class("pill")
        self.nut_bat_dau.set_sensitive(False)
        self.nut_bat_dau.connect("clicked", lambda *_: self._bat_dau())
        hop.append(self.nut_bat_dau)

        return cuon

    def _khi_ro_vao(self, *_):
        self.vung_tha.add_css_class("vung-tha-active")
        return Gdk.DragAction.COPY

    def _khi_ro_ra(self, *_):
        self.vung_tha.remove_css_class("vung-tha-active")

    # ------------------------------------------------------------ chọn đầu vào
    def _mo_chon_anh(self, *_):
        hop = Gtk.FileDialog(title="Chọn thư mục chứa ảnh")
        hop.select_folder(self, None, self._nhan_thu_muc)

    def _nhan_thu_muc(self, hop, ket_qua):
        try:
            tep = hop.select_folder_finish(ket_qua)
        except GLib.Error:
            return
        if tep:
            self._dat_thu_muc(Path(tep.get_path()))

    def _mo_chon_db(self, *_):
        loc = Gtk.FileFilter(name="Database COLMAP (*.db)")
        loc.add_pattern("*.db")
        kho_loc = Gio.ListStore.new(Gtk.FileFilter)
        kho_loc.append(loc)

        hop = Gtk.FileDialog(title="Chọn file database tải từ Colab")
        hop.set_filters(kho_loc)
        hop.set_default_filter(loc)
        hop.open(self, None, self._nhan_db)

    def _nhan_db(self, hop, ket_qua):
        try:
            tep = hop.open_finish(ket_qua)
        except GLib.Error:
            return
        if tep:
            self._dat_db(Path(tep.get_path()))

    def _mo_chon_dich(self, *_):
        hop = Gtk.FileDialog(title="Chọn nơi lưu file .zip và thư mục kết quả")
        bat_dau = self.thu_muc_dich or (
            self.thu_muc_goc.parent if self.thu_muc_goc else None)
        if bat_dau and bat_dau.is_dir():
            hop.set_initial_folder(Gio.File.new_for_path(str(bat_dau)))
        hop.select_folder(self, None, self._nhan_dich)

    def _nhan_dich(self, hop, ket_qua):
        try:
            tep = hop.select_folder_finish(ket_qua)
        except GLib.Error:
            return
        if tep:
            self._dat_dich(Path(tep.get_path()))

    def _dat_dich(self, duong: Path):
        if not os.access(duong, os.W_OK):
            self._bao(f"Không có quyền ghi vào “{duong.name}”")
            return

        # Cấm chọn đích nằm trong thư mục ảnh. Thư mục ảnh được gắn vào
        # container ở chế độ chỉ đọc, nên COLMAP sẽ không ghi nổi một chữ vào
        # đó — mà lỗi ấy phải hai tiếng sau mới lòi ra.
        if self.thu_muc_anh and (duong == self.thu_muc_anh
                                 or self.thu_muc_anh in duong.parents):
            self._bao("Không lưu vào bên trong thư mục ảnh được — "
                      "thư mục đó chỉ được đọc, không được ghi")
            return

        self.thu_muc_dich = duong
        if self.thu_muc_goc is None:
            self.hang_ra.set_subtitle(f"{duong}\n(chờ chọn thư mục ảnh)")
        else:
            self._tinh_duong_ra()
        self._nhac_neu_thieu_cho(duong)
        self._cap_nhat_san_sang()

    def _nhac_neu_thieu_cho(self, dich: Path):
        """
        Nhắc trước nếu ổ đích sắp hết chỗ.

        Chỗ cần dùng khoảng gấp đôi cỡ bộ ảnh: một lần cho ảnh đã nắn méo, một
        lần nữa cho file .zip — cộng thêm bản chép của database. Thà biết ngay
        bây giờ còn hơn chạy hai tiếng rồi chết vì đầy ổ.
        """
        if self.thu_muc_anh is None:
            return
        try:
            co_anh = sum(f.stat().st_size for f in self.thu_muc_anh.iterdir()
                         if f.is_file() and f.suffix.lower() in DUOI_ANH)
            co_db = self.file_db.stat().st_size if self.file_db else 0
            con_trong = shutil.disk_usage(dich).free
        except OSError:
            return
        can = co_anh * 2 + co_db
        if con_trong < can:
            self._bao(f"Ổ này chỉ còn {con_trong / 1e9:.1f} GB, "
                      f"mà cần khoảng {can / 1e9:.1f} GB")

    def _khi_tha(self, _dt, gia_tri, _x, _y):
        """Thả gì cũng nhận: thư mục thì coi là ảnh, file .db thì coi là database."""
        self.vung_tha.remove_css_class("vung-tha-active")
        ds = gia_tri.get_files()
        if not ds:
            return False
        nhan_duoc = False
        for tep in ds:
            duong = Path(tep.get_path())
            if duong.is_dir():
                self._dat_thu_muc(duong)
                nhan_duoc = True
            elif duong.suffix.lower() == ".db":
                self._dat_db(duong)
                nhan_duoc = True
        if not nhan_duoc:
            self._bao("Chỉ nhận thư mục ảnh hoặc file có đuôi .db")
        return nhan_duoc

    def _dat_thu_muc(self, duong: Path):
        """
        Chấp nhận cả hai kiểu thư mục:
          - thư mục chứa ảnh trực tiếp
          - thư mục dự án đã có sẵn thư mục con "input"
        """
        anh = (duong / "input"
               if (duong / "input").is_dir() and dem_anh(duong / "input") > 0
               else duong)

        n = dem_anh(anh)
        if n == 0:
            self._bao(f"Không thấy ảnh nào trong “{anh.name}”")
            return

        self.thu_muc_goc, self.thu_muc_anh, self.so_anh = duong, anh, n
        # Đọc cỡ thật của từng tấm ngay lúc này. Chỉ động vào vài chục byte đầu
        # mỗi tệp nên không thấy chậm, mà lại là dữ kiện duy nhất bắt được vụ
        # đưa nhầm thư mục ảnh — xem _cap_nhat_san_sang.
        self.co_anh_thuc, _, self.so_anh_kho_doc = co_anh_trong_thu_muc(anh)
        self.hang_anh.set_subtitle(
            f"{anh}   ({n} ảnh, {ta_co_anh(self.co_anh_thuc)})")
        self._tinh_duong_ra()
        self._cap_nhat_san_sang()

    def _tinh_duong_ra(self):
        """
        Chốt xem kết quả sẽ nằm ở đâu.

        File .zip luôn mang đúng tên thư mục người dùng thả vào. Thư mục kết
        quả thì phải thêm đuôi "_3d", vì nó nằm cùng chỗ với thư mục ảnh —
        trùng tên là ghi đè lên ảnh gốc.
        """
        if self.thu_muc_goc is None:
            return
        ten = self.thu_muc_goc.name

        if self.thu_muc_dich is not None:
            noi = self.thu_muc_dich
            ra = noi / f"{ten}_3d"
        elif self.thu_muc_anh != self.thu_muc_goc:
            # Thư mục dự án đã có sẵn thư mục con "input": ghi thẳng vào đó,
            # đúng cấu trúc mà Gaussian Splatting quen dùng.
            noi, ra = self.thu_muc_goc.parent, self.thu_muc_goc
        else:
            noi = self.thu_muc_goc.parent
            ra = noi / f"{ten}_3d"

        self.thu_muc_ra = ra
        self.file_zip = noi / f"{ten}.zip"
        self.hang_ra.set_subtitle(
            f"{self.file_zip}\n(thư mục kèm theo: {ra})"
            + ("" if self.thu_muc_dich is not None
               else "\nbấm vào đây để đổi nơi lưu"))

    def _dat_db(self, duong: Path):
        thong_tin = doc_database(duong)
        if thong_tin is None:
            self._bao(f"“{duong.name}” không phải database của COLMAP")
            return
        so_anh_db, so_cap, co_db = thong_tin
        self.file_db = duong
        self.so_anh_db = so_anh_db
        self.co_anh_db = co_db
        cd = duong.stat().st_size / 1e9
        # Đổi dấu phân cách nghìn RIÊNG cho con số, không đổi cho cả câu: chuỗi
        # cỡ ảnh ngăn nhau bằng ", " nên thay tuốt là nó thành ". ".
        so_cap_chu = f"{so_cap:,}".replace(",", ".")
        self.hang_db.set_subtitle(
            f"{duong}\n{so_anh_db} ảnh · {so_cap_chu} cặp đã ghép · "
            f"{ta_co_anh(co_db)} · {cd:.1f} GB")
        self._cap_nhat_san_sang()

    def _cap_nhat_san_sang(self):
        """Bật nút Bắt đầu khi đủ đầu vào, và cảnh báo nếu hai bên không khớp."""
        co_anh = self.thu_muc_anh is not None
        co_db = self.file_db is not None

        # Nơi lưu chọn trước, thư mục ảnh thả sau, và hoá ra nơi lưu nằm lọt
        # trong thư mục ảnh — lúc chọn chưa biết được nên phải rà lại ở đây.
        if (co_anh and self.thu_muc_ra
                and self.thu_muc_anh in self.thu_muc_ra.parents):
            self.hang_uoc_luong.set_title("Nơi lưu nằm trong thư mục ảnh")
            self.hang_uoc_luong.set_subtitle(
                "Thư mục ảnh chỉ được đọc nên không ghi kết quả vào đó được. "
                "Bấm “Chọn nơi lưu…” để chỉ sang chỗ khác.")
            self.nut_bat_dau.set_sensitive(False)
            return

        # ------------------------------------------------------------------
        # Cỡ ảnh phải khớp với cỡ ghi trong database. Đây là cái bẫy nguy hiểm
        # nhất của cả quy trình, và trước đây chỉ có một dòng chữ trong README
        # canh nó.
        #
        # Bảng cameras của .db ghi thông số nội tại tính theo điểm ảnh của bộ
        # ảnh đã ghép ở chặng 1. Đưa cho chặng 2 bộ ảnh cỡ khác thì tiêu cự và
        # tâm quang học lệch đi đúng bằng tỉ lệ hai cỡ — image_undistorter vẫn
        # chạy hết, vẫn xuất ra model, không một dòng cảnh báo nào. Cái sai chỉ
        # lộ ra sau khi train xong ở chặng 3.
        #
        # Nên chỗ này DỪNG HẲN chứ không cảnh báo suông, và in ra cả hai con số
        # để biết mình đang cầm nhầm cái gì.
        # ------------------------------------------------------------------
        if co_anh and co_db and self.co_anh_thuc and self.co_anh_db:
            khop = self.co_anh_thuc & self.co_anh_db
            if not khop:
                self.hang_uoc_luong.set_title("Ảnh không đúng cỡ ghi trong database")
                self.hang_uoc_luong.set_subtitle(
                    f"database ghi {ta_co_anh(self.co_anh_db)}, "
                    f"còn thư mục ảnh là {ta_co_anh(self.co_anh_thuc)}.\n"
                    "Nhiều khả năng đây là ảnh gốc, hoặc thư mục thu nhỏ ở một "
                    "cỡ khác. Thông số camera trong database ghi theo điểm ảnh "
                    "của bộ ảnh đã ghép, nên chạy tiếp là ra một model sai mà "
                    "không có gì báo lỗi.\n"
                    "Không có đường chữa ở đây: hoặc tìm đúng thư mục ảnh đã "
                    "đưa lên Colab, hoặc ghép lại từ chặng 1 bằng bộ ảnh mới.")
                self.nut_bat_dau.set_sensitive(False)
                return
            if self.co_anh_thuc - self.co_anh_db:
                # Khớp một phần: có tấm đúng cỡ, có tấm không. Chạy được, nhưng
                # thư mục đang lẫn ảnh của hai bộ khác nhau.
                self._bao(f"Thư mục lẫn nhiều cỡ ảnh "
                          f"({ta_co_anh(self.co_anh_thuc)}) — database chỉ ghi "
                          f"{ta_co_anh(self.co_anh_db)}")

        if co_anh and co_db and self.co_anh_db and self.so_anh_kho_doc:
            # Không đọc nổi cỡ thì không kiểm tra được. Nói thẳng là "không
            # biết", đừng để người dùng tưởng đã có ai canh giúp.
            self._bao(f"{self.so_anh_kho_doc} tấm không đọc được cỡ ảnh "
                      f"(chỉ đọc được JPEG và PNG) — chỗ đó không kiểm tra được")

        if co_anh and co_db and self.so_anh != self.so_anh_db:
            # Đây là cái bẫy dễ vấp nhất: lấy nhầm database của bộ ảnh khác.
            self.hang_uoc_luong.set_title(
                f"Lệch số ảnh: thư mục có {self.so_anh}, "
                f"database có {self.so_anh_db}")
            self.hang_uoc_luong.set_subtitle(
                "Nhiều khả năng database này của bộ ảnh khác. Chạy vẫn được "
                "nhưng ảnh thừa sẽ bị bỏ qua.")
        elif co_anh and co_db:
            phut = round(MOC_PHUT * self.so_anh / MOC_ANH)
            self.hang_uoc_luong.set_title(f"Ước chừng {phut} phút")
            self.hang_uoc_luong.set_subtitle(
                "Con số này chỉ là ước lượng dựa trên lần đo trước — "
                "thực tế thường lâu hơn chứ hiếm khi nhanh hơn.")
        else:
            thieu = []
            if not co_anh:
                thieu.append("thư mục ảnh")
            if not co_db:
                thieu.append("file .db")
            self.hang_uoc_luong.set_title(f"Còn thiếu: {' và '.join(thieu)}")
            self.hang_uoc_luong.set_subtitle("")

        self.nut_bat_dau.set_sensitive(co_anh and co_db)
        if co_anh and co_db:
            self.nhan_tha.set_label("Đủ rồi — bấm Bắt đầu")
            self.nhan_phu.set_label("thả thứ khác vào để đổi")
        elif co_anh:
            self.nhan_tha.set_label(f"{self.so_anh} ảnh — còn thiếu file .db")
        elif co_db:
            self.nhan_tha.set_label("Có database — còn thiếu thư mục ảnh")

    def _bao(self, chu: str):
        self.toast.add_toast(Adw.Toast(title=chu, timeout=5))

    # --------------------------------------------------------- trang lúc chạy
    def _trang_chay(self):
        cuon = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER)
        hop = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16,
                      margin_top=20, margin_bottom=20,
                      margin_start=20, margin_end=20)
        cuon.set_child(hop)

        cb = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        cb.add_css_class("canh-bao")
        n1 = Gtk.Label(label="ĐỪNG TẮT MÁY")
        n1.add_css_class("canh-bao-to")
        n2 = Gtk.Label(
            label="Máy đang dựng mô hình 3D. Tắt máy, sập nguồn hay đóng cửa sổ "
                  "là mất sạch, phải làm lại từ đầu.",
            wrap=True, justify=Gtk.Justification.CENTER)
        n2.add_css_class("canh-bao-nho")
        cb.append(n1)
        cb.append(n2)
        hop.append(cb)

        self.nhan_dong_ho = Gtk.Label(label="0:00", halign=Gtk.Align.CENTER)
        self.nhan_dong_ho.add_css_class("dong-ho")
        hop.append(self.nhan_dong_ho)

        self.nhan_con_lai = Gtk.Label(label="đang tính…", halign=Gtk.Align.CENTER)
        self.nhan_con_lai.add_css_class("con-lai")
        self.nhan_con_lai.add_css_class("nhat")
        hop.append(self.nhan_con_lai)

        hop_tong = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4,
                           margin_top=6)
        nhan = Gtk.Label(label="Toàn bộ công việc", halign=Gtk.Align.START)
        nhan.add_css_class("caption")
        nhan.add_css_class("nhat")
        hop_tong.append(nhan)
        self.thanh_tong = Gtk.ProgressBar(show_text=True)
        hop_tong.append(self.thanh_tong)
        hop.append(hop_tong)

        hop_chang = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.nhan_chang = Gtk.Label(label="Đang chuẩn bị…", halign=Gtk.Align.START)
        self.nhan_chang.add_css_class("heading")
        self.nhan_chang.set_wrap(True)
        hop_chang.append(self.nhan_chang)
        self.thanh_chang = Gtk.ProgressBar(show_text=True)
        hop_chang.append(self.thanh_chang)

        # Bằng chứng còn sống. COLMAP có những quãng im lặng dài (nắn chùm tia),
        # lúc ấy thanh tiến độ đứng yên và trông y hệt như treo. Dòng này nói rõ
        # container còn chạy hay không, để khỏi phải đoán qua mức dùng CPU.
        self.nhan_song = Gtk.Label(label="", halign=Gtk.Align.START, wrap=True)
        self.nhan_song.add_css_class("caption")
        self.nhan_song.add_css_class("nhat")
        hop_chang.append(self.nhan_song)
        hop.append(hop_chang)

        self.nhom_chang = Adw.PreferencesGroup(margin_top=6)
        self.hang_chang = {}
        for ma, ten, _ in CHANG:
            hang = Adw.ActionRow(title=ten)
            icon = Gtk.Image.new_from_icon_name("media-playback-stop-symbolic")
            icon.add_css_class("nhat-hon")
            hang.add_prefix(icon)
            nhan_tg = Gtk.Label(label="")
            nhan_tg.add_css_class("nhat")
            hang.add_suffix(nhan_tg)
            hang.add_css_class("nhat-hon")
            self.nhom_chang.add(hang)
            self.hang_chang[ma] = (hang, icon, nhan_tg)
        hop.append(self.nhom_chang)

        mo_rong = Gtk.Expander(label="Nhật ký chi tiết")
        cuon_log = Gtk.ScrolledWindow(min_content_height=160, vexpand=True)
        cuon_log.add_css_class("log")
        self.o_log = Gtk.TextView(editable=False, cursor_visible=False,
                                  monospace=True, wrap_mode=Gtk.WrapMode.CHAR)
        cuon_log.set_child(self.o_log)
        mo_rong.set_child(cuon_log)
        hop.append(mo_rong)

        self.nut_huy = Gtk.Button(label="Huỷ bỏ", halign=Gtk.Align.CENTER)
        self.nut_huy.add_css_class("destructive-action")
        self.nut_huy.add_css_class("pill")
        self.nut_huy.connect("clicked", lambda *_: self._hoi_huy())
        hop.append(self.nut_huy)

        return cuon

    # ------------------------------------------------------------------ chạy
    def _bat_dau(self):
        if not GOI_COLMAP.exists():
            self._bao(f"Thiếu gói COLMAP tại {GOI_COLMAP}")
            return
        if shutil.which("docker") is None:
            self._bao("Không tìm thấy Docker trên máy")
            return
        try:
            self.thu_muc_ra.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._bao(f"Không tạo được thư mục kết quả: {e}")
            return

        self.dang_chay = True
        self.dang_nen = False
        self.bi_huy = False
        self.luc_bat_dau = time.monotonic()
        self.luc_co_tin = self.luc_bat_dau
        self.chang_hien_tai = None
        self.phan_tram_chang = 0.0
        self.trong_so_da_qua = 0

        self.o_log.get_buffer().set_text("")
        for hang, icon, tg in self.hang_chang.values():
            icon.set_from_icon_name("media-playback-stop-symbolic")
            icon.remove_css_class("success")
            icon.add_css_class("nhat-hon")
            hang.add_css_class("nhat-hon")
            tg.set_label("")
        self.thanh_tong.set_fraction(0)
        self.thanh_tong.set_text("0%")
        self.thanh_chang.set_fraction(0)
        self.thanh_chang.set_text("")
        self.nhan_con_lai.set_label("đang tính…")
        self.nhan_song.set_label("đang khởi động container…")

        self._sang_trang("chay")
        self._an_nut_tieu_de(True)

        # Từ đây trở đi màn hình đã nói "ĐỪNG TẮT MÁY". Nếu có bất cứ trục trặc
        # nào mà ta để lọt, người dùng sẽ ngồi canh một màn hình đứng yên hàng
        # tiếng đồng hồ trong khi thật ra chẳng có gì chạy cả. Nên phải tự tay
        # bắt lỗi ở đây: hỏng thì quay về trang chờ và nói thẳng ra hỏng vì sao.
        try:
            # Chặn máy tự ngủ hoặc tự treo giữa chừng — đúng tinh thần
            # "đừng tắt máy". Không khoá được thì vẫn chạy tiếp, chỉ ghi lại.
            try:
                self.khoa_ngu = self.app.inhibit(
                    self,
                    Gtk.ApplicationInhibitFlags.SUSPEND |
                    Gtk.ApplicationInhibitFlags.IDLE |
                    Gtk.ApplicationInhibitFlags.LOGOUT,
                    "Đang dựng mô hình 3D")
            except Exception as e:                                # noqa: BLE001
                self.khoa_ngu = None
                self._ghi_log(f"(không khoá được chế độ ngủ: {e})")

            self.id_dong_ho = GLib.timeout_add_seconds(1, self._nhip_dong_ho)
            threading.Thread(target=self._chay_nen, daemon=True).start()
        except Exception as e:                                    # noqa: BLE001
            self._ket_thuc(1, f"không khởi động được: {e}")

    def _an_nut_tieu_de(self, an: bool):
        """
        Ẩn/hiện nút thu nhỏ - phóng to - đóng trên thanh tiêu đề.

        Adw.HeaderBar KHÔNG có set_show_title_buttons() như Gtk.HeaderBar —
        nó tách làm hai đầu. Gọi nhầm tên là ném AttributeError giữa chừng.
        """
        self.header.set_show_start_title_buttons(not an)
        self.header.set_show_end_title_buttons(not an)
        # Đang dựng mô hình mà bấm nén ảnh nữa thì bốn nhân chia đôi, cả hai
        # việc cùng chậm — giấu nút đi cho khỏi lỡ tay.
        self.nut_nen_anh.set_visible(not an)

    def _dung_lenh_docker(self):
        """
        Dựng lệnh docker, liệu cơm gắp mắm theo loại ổ đang dùng.

        Bình thường mỗi mount gắn thêm chữ "z" để docker dán nhãn SELinux cho
        thư mục, nhờ vậy container đụng vào được mà máy vẫn giữ nguyên hàng rào
        bảo vệ. Nhưng ổ exFAT/NTFS không giữ nổi nhãn, dán cũng như không, và
        container sẽ bị chặn ngay ở bước đọc thư mục.

        Gặp trường hợp đó thì bỏ chữ "z" đi và tắt hàng rào SELinux cho riêng
        lần chạy này. Đây là cách duy nhất ngoài việc chép hàng chục GB ảnh
        sang ổ trong rồi chép ngược lại.
        """
        cho_gan = [GOI_COLMAP.parent, self.thu_muc_anh,
                   self.file_db, self.thu_muc_ra]
        kho_tinh = [d for d in cho_gan
                    if loai_he_thong_tep(d) in HE_TEP_KHONG_NHAN]

        if kho_tinh:
            # ",z" đi kèm ":ro", còn ":z" đứng một mình khi mount đọc-ghi
            chi_doc, doc_ghi, them = ":ro", "", ["--security-opt",
                                                 "label=disable"]
            ten_o = ", ".join(sorted({loai_he_thong_tep(d) for d in kho_tinh}))
            GLib.idle_add(
                self._ghi_log,
                f"(ổ định dạng {ten_o} không giữ được nhãn SELinux — "
                f"tắt kiểm soát SELinux cho riêng container này)")
        else:
            chi_doc, doc_ghi, them = ":ro,z", ":z", []

        return [
            "docker", "run", "--rm", "-i", "--name", TEN_CONTAINER,
            *them,
            "-e", f"HOST_UID={os.getuid()}",
            "-e", f"HOST_GID={os.getgid()}",
            "-v", f"{GOI_COLMAP.parent}:/pkg{chi_doc}",
            "-v", f"{self.thu_muc_anh}:/in{chi_doc}",
            "-v", f"{self.file_db}:/db/database.db{chi_doc}",
            "-v", f"{self.thu_muc_ra}:/out{doc_ghi}",
            IMAGE_DOCKER, "bash", "-s",
        ]

    def _chay_nen(self):
        lenh = self._dung_lenh_docker()
        GLib.idle_add(self._ghi_log, "$ " + " ".join(lenh))
        try:
            self.tien_trinh = subprocess.Popen(
                lenh, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            self.tien_trinh.stdin.write(KICH_BAN)
            self.tien_trinh.stdin.close()
            for dong in self.tien_trinh.stdout:
                GLib.idle_add(self._dong_moi, dong.rstrip("\n"))
            ma = self.tien_trinh.wait()
        except Exception as e:                                    # noqa: BLE001
            GLib.idle_add(self._ket_thuc, 1, str(e))
            return
        GLib.idle_add(self._xong_container, ma)

    # Các mẫu chữ COLMAP in ra, lấy thẳng từ mã nguồn COLMAP 3.13.0:
    #   incremental_pipeline.cc : "Registering image #%d (num_reg_frames=%d)"
    #   undistortion.cc         : "Undistorting image [%d/%d]"
    RE_DUNG_CAM = re.compile(r"num_reg_frames=(\d+)")
    RE_CAP = re.compile(r"\[(\d+)\s*/\s*(\d+)\]")

    def _ghi_log(self, dong: str):
        """Thêm một dòng vào nhật ký chi tiết. Chỉ gọi từ luồng giao diện."""
        bo = self.o_log.get_buffer()
        bo.insert(bo.get_end_iter(), dong + "\n")
        if bo.get_line_count() > 3000:          # cắt bớt cho khỏi phình bộ nhớ
            bo.delete(bo.get_start_iter(), bo.get_iter_at_line(1000)[1])
        # Dùng scroll_to_iter chứ không tạo mark mới. Tạo mark cho từng dòng sẽ
        # để lại hàng chục nghìn mark trong bộ nhớ sau vài tiếng chạy.
        self.o_log.scroll_to_iter(bo.get_end_iter(), 0, False, 0, 0)
        return False

    def _dong_moi(self, dong: str):
        self._ghi_log(dong)
        self.luc_co_tin = time.monotonic()

        if dong.startswith("##CHANG:"):
            self._sang_chang(dong.split(":", 1)[1])
        elif dong.startswith("##LOI:"):
            self._bao(dong.split(":", 1)[1])
        elif dong.startswith("##XONG"):
            self._sang_chang(None)
        else:
            self._doc_tien_do(dong)
        return False

    def _doc_tien_do(self, dong: str):
        """Đọc tiến độ trong một dòng log của COLMAP, tuỳ theo chặng đang chạy."""
        if self.chang_hien_tai == "dung_cam":
            m = self.RE_DUNG_CAM.search(dong)
            if m and self.so_anh:
                da = int(m.group(1))
                self._dat_tien_do_chang(da / self.so_anh,
                                        f"{da} / {self.so_anh} ảnh đã dựng")
            return
        m = self.RE_CAP.search(dong)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            if y:
                self._dat_tien_do_chang(x / y, f"{x} / {y}")

    def _dat_tien_do_chang(self, phan: float, chu: str):
        self.phan_tram_chang = min(max(phan, 0.0), 1.0)
        self.thanh_chang.set_fraction(self.phan_tram_chang)
        self.thanh_chang.set_text(f"{chu}   ·   {self.phan_tram_chang * 100:.0f}%")
        self._cap_nhat_tong()

    def _cap_nhat_tong(self):
        """Ghép tiến độ chặng hiện tại vào tiến độ toàn bộ, theo tỉ trọng."""
        ts = TRONG_SO.get(self.chang_hien_tai, 0)
        xong = self.trong_so_da_qua + ts * self.phan_tram_chang
        phan = min(xong / TONG_TRONG_SO, 1.0)
        self.thanh_tong.set_fraction(phan)
        self.thanh_tong.set_text(f"{phan * 100:.0f}%")

        troi = time.monotonic() - self.luc_bat_dau
        if phan > 0.02 and troi > 20:
            self.nhan_con_lai.set_label(doc_con_lai(troi * (1 - phan) / phan))

    def _sang_chang(self, ma: str | None):
        gio = time.monotonic()
        if self.chang_hien_tai:
            hang, icon, tg = self.hang_chang[self.chang_hien_tai]
            icon.set_from_icon_name("object-select-symbolic")
            icon.remove_css_class("nhat-hon")
            icon.add_css_class("success")
            hang.remove_css_class("nhat-hon")
            tg.set_label(doc_thoi_gian(gio - self.luc_bat_dau_chang))
            self.trong_so_da_qua += TRONG_SO[self.chang_hien_tai]

        self.chang_hien_tai = ma
        self.luc_bat_dau_chang = gio
        self.phan_tram_chang = 0.0
        self.thanh_chang.set_fraction(0)
        self.thanh_chang.set_text("")

        if ma:
            hang, icon, _ = self.hang_chang[ma]
            hang.remove_css_class("nhat-hon")
            icon.set_from_icon_name("media-playback-start-symbolic")
            icon.remove_css_class("nhat-hon")
            self.nhan_chang.set_label(
                "Dựng vị trí camera — càng về cuối càng chậm"
                if ma == "dung_cam" else TEN_CHANG[ma])
        self._cap_nhat_tong()

    def _nhip_dong_ho(self):
        if not self.dang_chay:
            return False
        gio = time.monotonic()
        self.nhan_dong_ho.set_label(doc_thoi_gian(gio - self.luc_bat_dau))

        # Báo container còn sống hay không. Đây là chỗ duy nhất trả lời được
        # câu "nó có đang chạy thật không, hay chỉ đứng hình?" mà không bắt
        # người dùng đi mở trình theo dõi tài nguyên để đoán qua mức CPU.
        tt = self.tien_trinh
        if self.dang_nen:
            self.nhan_song.set_label(
                "container xong rồi · đang gói file .zip, đừng đụng vào thư mục")
        elif tt is None:
            self.nhan_song.set_label("đang khởi động container…")
        elif tt.poll() is not None:
            self.nhan_song.set_label("container đã dừng — đang thu dọn…")
        else:
            im = gio - self.luc_co_tin
            self.nhan_song.set_label(
                f"container đang chạy · tin mới nhất {doc_thoi_gian(im)} trước"
                + ("  (COLMAP hay im lặng lâu ở khúc nắn chùm tia — bình thường)"
                   if im > 120 else ""))
        return True

    # -------------------------------------------------------------------- huỷ
    def _hoi_huy(self):
        hop = Adw.AlertDialog(
            heading="Huỷ việc đang chạy?",
            body="Toàn bộ tiến độ từ đầu tới giờ sẽ mất, phải làm lại từ đầu.")
        hop.add_response("khong", "Chạy tiếp")
        hop.add_response("co", "Huỷ bỏ")
        hop.set_response_appearance("co", Adw.ResponseAppearance.DESTRUCTIVE)
        hop.set_default_response("khong")
        hop.connect("response", lambda _d, r: self._huy() if r == "co" else None)
        hop.present(self)

    def _huy(self):
        self.bi_huy = True
        self.nut_huy.set_sensitive(False)
        self.nut_huy.set_label("Đang dừng…")
        subprocess.run(["docker", "kill", TEN_CONTAINER],
                       capture_output=True, check=False)

    # ------------------------------------------------------------------- nén
    def _xong_container(self, ma: int):
        """
        Container đã đóng. Chạy trót lọt thì nén tiếp, còn không thì dừng luôn.

        Nén ở đây chứ không nén trong container, vì trong container không có
        sẵn lệnh zip, mà cài thêm thì phải tải gói về giữa chừng.
        """
        if ma != 0 or self.bi_huy:
            self._ket_thuc(ma, None)
            return False
        if not (self.thu_muc_ra / "sparse/0").is_dir():
            # Không có sparse/0 thì chẳng có gì đáng mang đi train. Vẫn coi là
            # xong để hộp thoại nói rõ chuyện này ra.
            self._ket_thuc(0, None)
            return False

        self.dang_nen = True
        self._sang_chang("nen_zip")
        self.nhan_chang.set_label("Nén thành file .zip để mang đi train")
        threading.Thread(target=self._luong_nen_zip, daemon=True).start()
        return False

    def _luong_nen_zip(self):
        """Gói images/ và sparse/ thành một file .zip duy nhất."""
        goc = self.thu_muc_ra
        ds = [f for ten in THU_MUC_MANG_DI
              for f in sorted((goc / ten).rglob("*")) if f.is_file()]
        tong_byte = sum(f.stat().st_size for f in ds) or 1

        # Ghi ra tên tạm rồi mới đổi tên. Huỷ giữa chừng hay mất điện sẽ để lại
        # file .dang-ghi thấy rõ là dở dang, chứ không phải một file .zip trông
        # như đã xong mà thật ra thiếu mất nửa số ảnh.
        tam = self.file_zip.with_name(self.file_zip.name + ".dang-ghi")
        ten_trong_zip = self.file_zip.stem
        da_byte, lan_bao = 0, 0.0
        try:
            with zipfile.ZipFile(tam, "w", zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as zf:
                for f in ds:
                    if self.bi_huy:
                        raise InterruptedError
                    # Ảnh JPEG/PNG đã nén sẵn trong ruột rồi, ép nén lần nữa chỉ
                    # tốn thêm hàng chục phút CPU mà tệp chẳng nhỏ đi được mấy.
                    kieu = (zipfile.ZIP_STORED if f.suffix.lower() in DUOI_ANH
                            else zipfile.ZIP_DEFLATED)
                    zf.write(f, f"{ten_trong_zip}/{f.relative_to(goc)}",
                             compress_type=kieu)
                    da_byte += f.stat().st_size
                    gio = time.monotonic()
                    if gio - lan_bao > 0.3:          # đừng dội quá nhiều vào GUI
                        lan_bao = gio
                        GLib.idle_add(self._tien_do_nen, da_byte, tong_byte)
        except InterruptedError:
            tam.unlink(missing_ok=True)
            GLib.idle_add(self._ket_thuc, 1, None)
            return
        except Exception as e:                                    # noqa: BLE001
            tam.unlink(missing_ok=True)
            GLib.idle_add(self._ket_thuc, 1, f"nén thất bại: {e}")
            return

        try:
            tam.replace(self.file_zip)
        except OSError as e:
            GLib.idle_add(self._ket_thuc, 1, f"không đổi tên được file zip: {e}")
            return
        GLib.idle_add(self._ket_thuc, 0, None)

    def _tien_do_nen(self, da: int, tong: int):
        self._dat_tien_do_chang(da / tong, f"{da / 1e9:.1f} / {tong / 1e9:.1f} GB")
        return False

    # --------------------------------------------------------------- kết thúc
    def _ket_thuc(self, ma: int, loi: str | None):
        self.dang_chay = False
        self.dang_nen = False
        if self.id_dong_ho:
            GLib.source_remove(self.id_dong_ho)
            self.id_dong_ho = None
        if self.khoa_ngu:
            self.app.uninhibit(self.khoa_ngu)
            self.khoa_ngu = None
        self.tien_trinh = None

        self._an_nut_tieu_de(False)
        self._sang_trang("cho")
        self.nut_huy.set_sensitive(True)
        self.nut_huy.set_label("Huỷ bỏ")

        tong = doc_thoi_gian(time.monotonic() - self.luc_bat_dau)
        if self.bi_huy:
            self._bao(f"Đã huỷ sau {tong}")
        elif ma == 0:
            self._hop_xong(tong)
        else:
            self._bao(f"Hỏng sau {tong} — mở “Nhật ký chi tiết” để xem lý do"
                      + (f": {loi}" if loi else ""))

    def _hop_xong(self, tong: str):
        if self.file_zip.is_file():
            cd = self.file_zip.stat().st_size / 1e9
            than = (f"Mất {tong}.\n\nFile mang đi train:\n{self.file_zip}"
                    f"   ({cd:.1f} GB)\n\n"
                    "Đẩy nguyên file này lên là train được — bên trong đã có "
                    "images/ và sparse/0 đúng cấu trúc Gaussian Splatting.")
        else:
            than = (f"Mất {tong}.\n\nKết quả nằm ở:\n{self.thu_muc_ra}\n\n"
                    "Nhưng không thấy thư mục sparse/0 nên chưa nén được — "
                    "xem lại nhật ký.")
        hop = Adw.AlertDialog(heading="Xong rồi", body=than)
        hop.add_response("dong", "Đóng")
        hop.add_response("mo", "Mở thư mục")
        hop.set_default_response("mo")
        hop.connect("response", self._tra_loi_xong)
        hop.present(self)

    def _tra_loi_xong(self, _hop, tra_loi):
        if tra_loi != "mo":
            return
        # Mở thư mục chứa file .zip, chứ không mở thư mục kết quả: thứ người
        # dùng cần cầm đi lúc này là cái file zip. Chỉ khi chưa nén được mới
        # quay về cách cũ.
        if self.file_zip.is_file():
            try:
                Gio.AppInfo.launch_default_for_uri(
                    Gio.File.new_for_path(str(self.file_zip.parent)).get_uri(),
                    None)
                return
            except GLib.Error:
                pass
        Gio.AppInfo.launch_default_for_uri(
            Gio.File.new_for_path(str(self.thu_muc_ra)).get_uri(), None)

    def _khi_dong_cua_so(self, *_):
        if self.trang_nen.dang_chay:
            self.trang_nen.bi_huy = True     # dừng nén rồi hẵng đóng
        if not self.dang_chay:
            return False
        hop = Adw.AlertDialog(
            heading="Đang chạy dở",
            body="Đóng cửa sổ bây giờ sẽ huỷ toàn bộ tiến độ. Chắc chưa?")
        hop.add_response("khong", "Để chạy tiếp")
        hop.add_response("co", "Đóng và huỷ")
        hop.set_response_appearance("co", Adw.ResponseAppearance.DESTRUCTIVE)
        hop.set_default_response("khong")
        hop.connect(
            "response",
            lambda _d, r: (self._huy(), self.destroy()) if r == "co" else None)
        hop.present(self)
        return True          # chặn, không cho đóng cửa sổ ngay


class Ung(Adw.Application):
    def __init__(self):
        super().__init__(application_id="io.github.ryanhuhut.quet3d")

    def _loi_lot_luoi(self, kieu, gia_tri, vet):
        """
        Lưới an toàn cuối cùng.

        Khi một hàm phản hồi của GTK ném lỗi, PyGObject chỉ in vết lỗi ra stderr
        rồi chạy tiếp như không có chuyện gì. Mở app từ menu thì stderr đổ vào
        journald — người dùng không thấy gì hết, chỉ thấy giao diện đứng im.
        Chính kiểu lỗi đó đã làm màn hình "ĐỪNG TẮT MÁY" hiện ra mà bên dưới
        không có gì chạy. Nên bắt lại và nói ra màn hình.
        """
        traceback.print_exception(kieu, gia_tri, vet)     # vẫn ghi vào journal
        cua = self.props.active_window
        if cua is not None:
            GLib.idle_add(cua._bao, f"Lỗi trong tool — {kieu.__name__}: {gia_tri}")

    def do_startup(self):
        Adw.Application.do_startup(self)
        sys.excepthook = self._loi_lot_luoi
        nha_cc = Gtk.CssProvider()
        nha_cc.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), nha_cc,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

    def do_activate(self):
        cua = self.props.active_window or CuaSo(self)
        cua.present()


if __name__ == "__main__":
    sys.exit(Ung().run(sys.argv))
