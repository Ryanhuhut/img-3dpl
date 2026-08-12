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

import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
GOI_COLMAP = Path.home() / ".local/share/quet3d/colmap.tar.gz"
IMAGE_DOCKER = "ubuntu:22.04"
TEN_CONTAINER = "quet3d-dang-chay"

DUOI_ANH = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Mã chặng, tên hiển thị, tỉ trọng thời gian.
# Tỉ trọng lấy từ số đo thật trên máy Acer i3 với 320 ảnh:
# dựng camera 105 phút, các chặng còn lại chưa tới 1 phút mỗi chặng.
CHANG = [
    ("chuan_bi", "Chuẩn bị",             2),
    ("dung_cam", "Dựng vị trí camera", 105),
    ("kiem_tra", "Kiểm tra kết quả",     1),
    ("nan_meo",  "Nắn méo ảnh",          2),
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
"""


def dem_anh(thu_muc: Path) -> int:
    """Đếm số file ảnh nằm trực tiếp trong thư mục."""
    try:
        return sum(1 for f in thu_muc.iterdir()
                   if f.is_file() and f.suffix.lower() in DUOI_ANH)
    except OSError:
        return 0


def doc_database(duong: Path):
    """
    Đọc thử file database của COLMAP.
    Trả về (số ảnh, số cặp đã ghép) hoặc None nếu file không phải database COLMAP.
    """
    try:
        with sqlite3.connect(f"file:{duong}?mode=ro", uri=True) as d:
            c = d.cursor()
            so_anh = c.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            so_cap = c.execute(
                "SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
            return so_anh, so_cap
    except sqlite3.Error:
        return None


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


class CuaSo(Adw.ApplicationWindow):

    def __init__(self, app):
        super().__init__(application=app, title="Quét 3D")
        self.set_default_size(800, 800)

        self.app = app
        self.thu_muc_anh: Path | None = None
        self.thu_muc_ra: Path | None = None
        self.file_db: Path | None = None
        self.so_anh = 0
        self.so_anh_db = 0

        self.dang_chay = False
        self.bi_huy = False
        self.tien_trinh: subprocess.Popen | None = None
        self.luc_bat_dau = 0.0
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
        khung.add_top_bar(self.header)
        self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        khung.set_content(self.stack)
        self.toast.set_child(khung)

        self.stack.add_named(self._trang_cho(), "cho")
        self.stack.add_named(self._trang_chay(), "chay")

        self.connect("close-request", self._khi_dong_cua_so)

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
        self.hang_ra = Adw.ActionRow(title="Kết quả sẽ ghi vào", subtitle="—")
        self.hang_ra.add_prefix(Gtk.Image.new_from_icon_name("folder-symbolic"))
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
        if (duong / "input").is_dir() and dem_anh(duong / "input") > 0:
            anh, ra = duong / "input", duong
        else:
            anh, ra = duong, duong.parent / f"{duong.name}_3d"

        n = dem_anh(anh)
        if n == 0:
            self._bao(f"Không thấy ảnh nào trong “{anh.name}”")
            return

        self.thu_muc_anh, self.thu_muc_ra, self.so_anh = anh, ra, n
        self.hang_anh.set_subtitle(f"{anh}   ({n} ảnh)")
        self.hang_ra.set_subtitle(str(ra))
        self._cap_nhat_san_sang()

    def _dat_db(self, duong: Path):
        thong_tin = doc_database(duong)
        if thong_tin is None:
            self._bao(f"“{duong.name}” không phải database của COLMAP")
            return
        so_anh_db, so_cap = thong_tin
        self.file_db = duong
        self.so_anh_db = so_anh_db
        cd = duong.stat().st_size / 1e9
        self.hang_db.set_subtitle(
            f"{duong}\n{so_anh_db} ảnh · {so_cap:,} cặp đã ghép · {cd:.1f} GB"
            .replace(",", "."))
        self._cap_nhat_san_sang()

    def _cap_nhat_san_sang(self):
        """Bật nút Bắt đầu khi đủ đầu vào, và cảnh báo nếu hai bên không khớp."""
        co_anh = self.thu_muc_anh is not None
        co_db = self.file_db is not None

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
        self.bi_huy = False
        self.luc_bat_dau = time.monotonic()
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

        self.stack.set_visible_child_name("chay")
        self.header.set_show_title_buttons(False)

        # Chặn máy tự ngủ hoặc tự treo giữa chừng — đúng tinh thần "đừng tắt máy"
        self.khoa_ngu = self.app.inhibit(
            self,
            Gtk.ApplicationInhibitFlags.SUSPEND |
            Gtk.ApplicationInhibitFlags.IDLE |
            Gtk.ApplicationInhibitFlags.LOGOUT,
            "Đang dựng mô hình 3D")

        self.id_dong_ho = GLib.timeout_add_seconds(1, self._nhip_dong_ho)
        threading.Thread(target=self._chay_nen, daemon=True).start()

    def _chay_nen(self):
        lenh = [
            "docker", "run", "--rm", "-i", "--name", TEN_CONTAINER,
            "-e", f"HOST_UID={os.getuid()}",
            "-e", f"HOST_GID={os.getgid()}",
            "-v", f"{GOI_COLMAP.parent}:/pkg:ro,z",
            "-v", f"{self.thu_muc_anh}:/in:ro,z",
            "-v", f"{self.file_db}:/db/database.db:ro,z",
            "-v", f"{self.thu_muc_ra}:/out:z",
            IMAGE_DOCKER, "bash", "-s",
        ]
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
        GLib.idle_add(self._ket_thuc, ma, None)

    # Các mẫu chữ COLMAP in ra, lấy thẳng từ mã nguồn COLMAP 3.13.0:
    #   incremental_pipeline.cc : "Registering image #%d (num_reg_frames=%d)"
    #   undistortion.cc         : "Undistorting image [%d/%d]"
    RE_DUNG_CAM = re.compile(r"num_reg_frames=(\d+)")
    RE_CAP = re.compile(r"\[(\d+)\s*/\s*(\d+)\]")

    def _dong_moi(self, dong: str):
        bo = self.o_log.get_buffer()
        bo.insert(bo.get_end_iter(), dong + "\n")
        if bo.get_line_count() > 3000:          # cắt bớt cho khỏi phình bộ nhớ
            bo.delete(bo.get_start_iter(), bo.get_iter_at_line(1000)[1])
        # Dùng scroll_to_iter chứ không tạo mark mới. Tạo mark cho từng dòng sẽ
        # để lại hàng chục nghìn mark trong bộ nhớ sau vài tiếng chạy.
        self.o_log.scroll_to_iter(bo.get_end_iter(), 0, False, 0, 0)

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
        self.nhan_dong_ho.set_label(
            doc_thoi_gian(time.monotonic() - self.luc_bat_dau))
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

    # --------------------------------------------------------------- kết thúc
    def _ket_thuc(self, ma: int, loi: str | None):
        self.dang_chay = False
        if self.id_dong_ho:
            GLib.source_remove(self.id_dong_ho)
            self.id_dong_ho = None
        if self.khoa_ngu:
            self.app.uninhibit(self.khoa_ngu)
            self.khoa_ngu = None

        self.header.set_show_title_buttons(True)
        self.stack.set_visible_child_name("cho")
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
        co_sparse = (self.thu_muc_ra / "sparse/0").is_dir()
        hop = Adw.AlertDialog(
            heading="Xong rồi",
            body=f"Mất {tong}.\n\nKết quả nằm ở:\n{self.thu_muc_ra}\n\n"
                 + ("Thư mục sparse/0 và images/ đã sẵn sàng cho Gaussian Splatting."
                    if co_sparse
                    else "Nhưng không thấy thư mục sparse/0 — xem lại nhật ký."))
        hop.add_response("dong", "Đóng")
        hop.add_response("mo", "Mở thư mục")
        hop.set_default_response("mo")
        hop.connect("response", self._tra_loi_xong)
        hop.present(self)

    def _tra_loi_xong(self, _hop, tra_loi):
        if tra_loi == "mo":
            Gio.AppInfo.launch_default_for_uri(
                Gio.File.new_for_path(str(self.thu_muc_ra)).get_uri(), None)

    def _khi_dong_cua_so(self, *_):
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

    def do_startup(self):
        Adw.Application.do_startup(self)
        nha_cc = Gtk.CssProvider()
        nha_cc.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), nha_cc,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

    def do_activate(self):
        cua = self.props.active_window or CuaSo(self)
        cua.present()


if __name__ == "__main__":
    import sys
    sys.exit(Ung().run(sys.argv))
