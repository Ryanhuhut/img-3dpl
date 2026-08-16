# img-3dpl

***img**age **-** to **3d** **p**ipe**l**ine — quy trình biến ảnh thành 3D*

Biến một thư mục ảnh chụp thường thành mô hình Gaussian Splatting, trên máy tính
không có card đồ hoạ nào chạy được.

[English version → README.md](README.md)

**Phạm vi:** quy trình này quét **vật thể lẻ** — hộp bài, bìa sách, trang
truyện. Quét **cả căn phòng** thì đang làm dở và hiện bị khoá: chặng 3 xong
rồi, chặng 1 và 2 thì chưa. Chọn preset phòng sẽ báo lỗi ngay thay vì để bạn
mất năm tiếng lấy về một kết quả không dùng được. Chi tiết ở
[`docs/ROOM_MODE.md`](docs/ROOM_MODE.md).

---

## Thật sự không cần GPU

Toàn bộ quy trình này được dựng và thử nghiệm trên một chiếc laptop phổ thông có
card đồ hoạ rời **đã chết hẳn phần cứng** — Intel i3-1005G1, hai nhân, chỉ còn
đồ hoạ tích hợp. Nó vẫn biến 320 tấm ảnh thành mô hình 3D.

Ba điều làm nên chuyện đó:

- **Biên dịch mã CUDA không cần thiết bị CUDA.** Nó chỉ cần `nvcc`, thứ nằm sẵn
  trong một image Docker. Máy bạn không hề chạy một lệnh GPU nào để dựng ra phần
  mềm này.
- **Chặng ngốn GPU chạy trên T4 miễn phí của Colab.** Cài đặt ở đó mất bảy giây —
  không `apt install`, không `ldconfig`, không biên dịch gì.
- **Chặng ở lại máy bạn vốn không cần GPU.** Dựng vị trí camera là việc tuần tự.
  Nhân CPU thắng card đồ hoạ ở khoản này, mà Colab miễn phí chỉ cho hai nhân.

Laptop của bạn chạy được Docker và mở được trình duyệt là làm được. Chậm hơn máy
trạm, đúng. Nhưng vẫn xong.

---

Chỗ vướng khi làm việc này trên Colab: gói `colmap` cài bằng `apt` được **biên
dịch không kèm CUDA**. Ghép ảnh kiểu `exhaustive` cho 320 tấm chạy bằng CPU mất
hơn ba tiếng mà vẫn chưa xong. Kho này chữa đúng chỗ đó, đồng thời chia việc ra
sao cho mỗi chặng chạy ở nơi nó nhanh nhất.

---

## Chia việc thế nào

| Chặng | Chạy ở đâu | Vì sao ở đó | 320 ảnh |
|---|---|---|---|
| **1. Ghép ảnh** | Colab T4 | Việc chia nhỏ được cho hàng nghìn nhân, GPU thắng ~23 lần | **28 phút** |
| **2. Dựng vị trí camera** | Máy của bạn | Việc tuần tự — GPU bó tay, mà máy cá nhân thường nhiều nhân CPU hơn Colab miễn phí | ~105 phút |
| **3. Huấn luyện Gaussian Splatting** | Colab T4 | Cần CUDA | 45-70 phút |

Chặng 2 là chỗ nhiều người hiểu nhầm. Khâu tinh chỉnh có thể dùng GPU, nhưng
khâu định vị từng ảnh thì tuần tự về bản chất: ảnh mới phụ thuộc vào những ảnh
đã đặt trước đó. Colab bậc miễn phí chỉ cho 2 nhân CPU, còn máy tính cá nhân
thường có nhiều hơn.

```
 ảnh đã về 3200px ──►  [1] GPU Colab     ──►  Meo_3200.db
       dạng .zip            ghép các cặp

    Meo_3200.db   ──►  [2] app máy tính  ──►  Meo_3200_3d/
   + đúng ảnh đó           dựng camera          images/ + sparse/0/

   Meo_3200_3d/   ──►  [3] GPU Colab     ──►  Meo_3200.ply
       dạng .zip            huấn luyện
```

---

## Số đo thật

320 ảnh JPEG 1200×1600, cùng một máy điện thoại, chụp vòng quanh một vật thể.

| Lệnh | Máy | Số cặp phải so | Thời gian | Giây/cặp |
|---|---|---|---|---|
| `sequential_matcher` | Intel i3-1005G1, CPU | ~3.000 | 38,6 phút | 0,772 |
| `exhaustive_matcher` | CPU Colab (colmap từ `apt`) | 51.040 | hơn 3 giờ, **chưa từng xong** | — |
| `exhaustive_matcher` | **Colab T4, quy trình này** | **51.040** | **28 phút** | **0,033** |

Nhanh hơn khoảng **23 lần** tính trên cùng khối lượng việc. Nhưng con số đó chưa
phải điều đáng kể nhất — đáng kể là `exhaustive` chuyển từ *bất khả thi* sang
*chuyện thường*. Bắt con i3 chạy đúng 51.040 cặp ấy sẽ mất chừng 11 tiếng.

Ghép kiểu `exhaustive` còn cho mô hình **tốt hơn**. Kiểu `sequential` chỉ so mỗi
ảnh với 10 ảnh lân cận, nên khi bạn đi vòng quanh vật thể, nó không nhận ra ảnh
cuối chồng lấn với ảnh đầu. Bỏ sót chỗ khép vòng đó khiến mô hình bị trôi lệch.

---

## Bắt đầu nhanh

Bốn chặng, đánh số 0 đến 3 bên dưới. Ba chỗ nối giữa chúng là nơi người ta mất
toi cả buổi chiều:

| Chỗ nối | Hay sai thế nào | Làm đúng |
|---|---|---|
| 1 → 2 | Tải cả thư mục trên Drive về | Chỉ tải **mỗi file `.db`**. Ảnh đã nằm sẵn trên máy rồi — chính là thư mục đã thu nhỏ ở chặng 0 |
| 2 | Thả **ảnh gốc** vào app | Phải thả **thư mục ảnh đã thu nhỏ**, đúng cái đã đưa lên Colab. Tên file y hệt nhau nên không có lỗi nào báo cả, chỉ có mô hình dựng ra là sai |
| 0 → 1 | Tự giải nén, hoặc sắp lại zip cho "đúng chuẩn" | Không cần. Cả hai notebook tự tìm ảnh, và tự tìm `images/` + `sparse/0/`, nằm sâu mấy tầng cũng ra |

Và đặt tên cho từng dự án. File database được lưu thành `Meo_3200.db` chứ không
phải `database.db`, để hai dự án không bao giờ biến thành hai file trùng tên nằm
chung một thư mục Downloads.

### Chặng 0 — thu nhỏ ảnh trước đã

Làm trước tiên. Ghép ảnh 4000px không tốt hơn ghép ảnh nhỏ hơn, chỉ chậm hơn
thôi — COLMAP tự hạ về 3200px để dò đặc trưng. Nén trước cũng kéo dung lượng
phải tải lên từ vài GB xuống vài trăm MB.

**Chọn cỡ ảnh ở đây, không phải để lát nữa.** Chọn gì thì cỡ đó đi xuyên suốt
phần còn lại: database chặng 1 ghi thông số camera của đúng cỡ này, chặng 2 nắn
méo ra đúng cỡ này, chặng 3 train ở đúng cỡ này. Thêm cờ ở chặng 3 không gỡ lại
được quyết định đã chốt ở đây.

Chọn theo thứ mình chụp, đừng chọn theo số điểm ảnh. Tên preset ở đây đúng bằng
tên dùng ở chặng 3, nên chọn một lần là đi suốt:

| preset | cỡ | dùng khi | cái giá |
|---|---|---|---|
| **`FLAT_OBJECT`** | **3200px** — mặc định | vật thể có chữ nhỏ cần đọc được | gấp bốn số điểm ảnh, chặng 3 chậm khoảng ba lần, và RAM Colab free chỉ chứa nổi chừng 450 tấm ở cỡ này |
| **`COMPLEX_OBJECT`** | **2400px** | vật thể nhiều gờ cạnh, ít chữ | |
| **`ENTIRE_ROOM`** | **1600px** | cả căn phòng — mặc định cũ của mọi trường hợp | nhanh nhất ở mọi chặng |

Chữ cao 20px trên ảnh 3200px hạ về 1600px chỉ còn 10px — sát ngưỡng Nyquist, và
qua JPEG chất lượng 93 nữa thì gần như không còn gì. Mất ở chặng 0 thì không
tham số train nào lấy lại được. Đó là lý do 1600px thôi làm mặc định: nó chưa
bao giờ sai về hình khối, chỉ sai về chữ — mà chữ mới là thứ người ta tìm tới
đây để làm.

3200px là cái trần có thật chứ không phải con số cho tròn: COLMAP tự hạ ảnh
xuống `max_image_size` (mặc định 3200) để dò đặc trưng rồi mới nhân toạ độ
keypoint trở lại cỡ gốc. Ảnh to hơn 3200px là tốn công tải lên mà không thêm
được đặc trưng nào.

Lên cỡ lớn thì nên chụp ít ảnh đi: 180 ảnh 3200px về đích nhanh hơn 320 ảnh
1600px mà lại nét hơn, vì chặng 1 chỉ phải ghép một phần ba số cặp (16.110 thay
vì 51.040). 180 tấm quanh một vật thể là mỗi tấm cách nhau 2 độ — dư thừa so với
mức 5-10 độ mà việc dựng hình thật sự cần.

App máy tính làm sẵn việc này — nút **Nén ảnh** ở màn hình chính tự lấy cỡ ảnh
theo preset, thu nhỏ, kiểm tra EXIF, rồi gói lại thành một tệp. Đoạn script dưới
đây là làm tay đúng ngần ấy việc.

Cần ImageMagick 7 (lệnh `magick`). Sửa bốn dòng đầu, phần còn lại dán nguyên.
`CANH` là con số duy nhất đáng bận tâm — lấy theo bảng bên trên, và nó đặt luôn
tên thư mục đích để hai lần chạy khác cỡ không đè lên nhau:

```bash
# ==================== SUA O DAY ====================
CANH=3200                                        # 3200 FLAT_OBJECT / 2400 COMPLEX_OBJECT / 1600 ENTIRE_ROOM
NGUON="/home/ryanhuhut/Downloads/Cap-GB"        # thu muc anh goc — doi khi quet vat khac
DICH="/home/ryanhuhut/quet3d/Cap-GB_$CANH"      # thu muc dich — nen dat theo ten du an
DUOI="jpg"                                       # duoi anh: jpg / jpeg / png
# ===================================================

cd "$NGUON" || { echo "KHONG THAY THU MUC"; exit 1; }

echo "--- So anh: $(ls *.$DUOI 2>/dev/null | wc -l)"
echo "--- Tieu cu EXIF (phai ra MOT dong duy nhat):"
magick identify -format "%[EXIF:FocalLengthIn35mmFilm] " *.$DUOI 2>/dev/null | tr ' ' '\n' | sort | uniq -c

mkdir -p "$DICH"
echo "--- Dang thu nho ve ${CANH}px, doi vai phut..."
magick mogrify -path "$DICH" -resize "${CANH}x${CANH}" -quality 93 *.$DUOI

echo "--- XONG: $(ls "$DICH" | wc -l) anh, $(du -sh "$DICH" | cut -f1)"
```

Dòng EXIF mới là dòng quan trọng. **Nó phải in ra đúng một hàng.** Ra hai hàng
nghĩa là trong thư mục có hai tiêu cự khác nhau — zoom bị xê dịch, hoặc ảnh của
hai máy — và lúc đó một model camera không tả nổi tất cả. Hoặc bỏ mấy tấm lạc
loài đi, hoặc đặt `SINGLE_CAMERA = False` ở chặng 1.

Xong thì đóng gói thư mục vừa nén:

```bash
# ==================== SUA O DAY ====================
CANH=3200                                    # dung con so o lenh truoc
DICH="/home/ryanhuhut/quet3d/Cap-GB_$CANH"   # thu muc anh da thu nho — dung ten o lenh truoc
# ===================================================

cd "$(dirname "$DICH")" || exit 1
zip -r -0 "$(basename "$DICH").zip" "$(basename "$DICH")"
echo "--- XONG: $(du -sh "$(basename "$DICH").zip" | cut -f1)"
```

`-0` là chỉ gói lại, không nén. JPEG vốn đã nén rồi; bắt zip nén thêm lần nữa
tốn vài phút mà chẳng bớt được byte nào. Đưa file zip đó lên Drive.

Ảnh nằm ngay tầng gốc của zip hay nằm trong một thư mục con đều được — chặng 1
tự tìm ra, và tự bỏ qua `__MACOSX/` cùng các file ẩn.

### Chặng 1 — ghép ảnh, trên Colab

Mở [`notebooks/1_match_images_colab.ipynb`](notebooks/1_match_images_colab.ipynb)
bằng Colab, chọn `Runtime` → `Change runtime type` → **T4 GPU**, sửa ô cấu hình,
rồi chạy hết. Không phải tự giải nén gì cả — notebook tự làm, và tự tìm ra ảnh
dù chúng nằm ngay tầng gốc của zip hay nằm trong thư mục con.

Kết quả là file `Cap-GB_3200.db` nằm trên Drive, đặt tên theo file zip của bạn
chứ không phải `database.db` như mọi hướng dẫn khác. Chuyện tên gọi này thành
quan trọng ngay khi bạn có dự án thứ hai: hai file cùng tên `database.db` nằm
chung thư mục Downloads chính là cách file sai lọt vào chặng 2, và hai tiếng sau
bạn mới biết.

Cài COLMAP trong Colab mất **7 giây** — gói dựng sẵn, tự chứa, không cần
`apt install`, không cần `ldconfig`:

```python
!wget -q https://github.com/Ryanhuhut/colmap-cuda-colab/releases/latest/download/colmap-3.13.0-cuda12.2-ubuntu2204-sm75.tar.gz -O /tmp/colmap.tar.gz
!tar -xzf /tmp/colmap.tar.gz -C /opt
import os; os.environ["PATH"] = "/opt/colmap-cuda/bin:" + os.environ["PATH"]
!colmap -h | head -3
```

Dòng cuối phải in ra chữ `with CUDA`.

### Chặng 2 — dựng vị trí camera, trên máy bạn

Tải file `.db` từ Drive về — **chỉ mỗi file đó thôi.** Ảnh thì đã nằm sẵn trên
máy rồi: chính là thư mục đã thu nhỏ bạn tạo ở chặng 0. Rồi chạy app. Nó chỉ cần
Docker, không cần gì khác — COLMAP chạy bên trong container nên máy bạn không bị
cài thêm thứ gì.

```bash
git clone https://github.com/Ryanhuhut/img-3dpl
cd img-3dpl/desktop-app

mkdir -p ~/.local/share/quet3d
wget https://github.com/Ryanhuhut/colmap-cuda-colab/releases/latest/download/colmap-3.13.0-cuda12.2-ubuntu2204-sm75.tar.gz \
     -O ~/.local/share/quet3d/colmap.tar.gz

chmod +x quet3d.py
./quet3d.py
```

Thả thư mục ảnh và file `.db` vào, bấm Bắt đầu. App hiện đồng hồ đã chạy, thanh
tiến độ đọc trực tiếp từ output của COLMAP, ước lượng thời gian còn lại, và một
bảng cảnh báo đỏ to đùng để không ai lỡ tay tắt máy giữa chừng. Nó cũng chặn máy
tự ngủ trong lúc làm việc.

**Thư mục ảnh phải là đúng thư mục đã thu nhỏ và đưa lên Colab, không phải ảnh
gốc.** Tên file hai bên y hệt nhau nên chẳng có lỗi nào báo cả — nhưng thông số
camera nằm trong file `.db` mô tả đúng những tấm đã thu nhỏ ấy, đưa ảnh cỡ khác
vào thì COLMAP không chết, nó chỉ dựng ra một mô hình sai. Nhớ giữ thư mục đó
lại, đừng xoá sau khi đã đóng zip đưa lên Drive.

Chuyện này đúng với mọi cỡ ảnh, không riêng 1600px: chặng 0 chọn 3200px thì
chặng này phải nhận đúng thư mục 3200px.

Kết quả nằm cạnh thư mục ảnh, tên là `<tên thư mục>_3d/` — `Cap-GB_3200/` sẽ cho
ra `Cap-GB_3200_3d/`.

Cần GTK4 và libadwaita — hai thứ có sẵn trên mọi bản GNOME hiện hành
(`python3-gobject gtk4 libadwaita`).

### Chặng 3 — huấn luyện, trên Colab

Chặng 2 để lại cho bạn một thư mục như thế này:

```
Mèo_3d/
├── images/                      ← CẦN: ảnh đã nắn méo, tên y hệt ảnh gốc
├── sparse/0/
│   ├── cameras.bin              ← CẦN: thông số ống kính
│   ├── images.bin               ← CẦN: vị trí + hướng của từng ảnh (cái chạy 2 tiếng)
│   └── points3D.bin             ← CẦN: đám mây điểm thưa
├── distorted/
│   ├── database.db              ← không cần, và là file NẶNG NHẤT (vài GB)
│   └── sparse/0/*.bin           ← không cần: mô hình lúc chưa nắn méo
├── stereo/                      ← không cần: khung rỗng cho dựng dày
└── run-colmap-*.sh              ← không cần: script COLMAP tự sinh
```

Đóng gói cả cụm thành một file rồi tải lên:

```bash
# ==================== SUA O DAY ====================
CANH="/home/ryanhuhut/quet3d/Mèo_3d"    # thu muc chang 2 vua tao ra
# ===================================================

cd "$(dirname "$CANH")" || exit 1
TEN="$(basename "$CANH")"
zip -r -0 "${TEN%_3d}.zip" "$TEN" -x "$TEN/distorted/*" "$TEN/stereo/*"
echo "--- XONG: $(du -sh "${TEN%_3d}.zip" | cut -f1)"
```

Hai cái `-x` bỏ `distorted/` và `stereo/` ra ngoài. Có để lại thì notebook cũng
tự bỏ qua, nhưng `database.db` thường nặng vài GB — tải lên là mất đứt một tiếng
đồng hồ không đổi lại được gì.

Rồi mở
[`notebooks/3_train_gaussian_splatting_colab.ipynb`](notebooks/3_train_gaussian_splatting_colab.ipynb),
dán đường dẫn file zip đó vào **ô số 6** và chạy hết. Notebook tự tìm `images/`
với `sparse/0/` nằm ở tầng nào cũng ra, chỉ giải nén đúng hai thứ đó, tự đặt tên
kết quả theo tên file zip (`Mèo.zip` → `Meo_30000.ply`), rồi train. Đưa
`.tar.gz` hay để nguyên thư mục trên Drive cũng chạy y như vậy.

Thêm một dòng nữa trong ô 6 chọn bộ tham số train:

```python
PRESET = "FLAT_OBJECT"   # "FLAT_OBJECT" | "COMPLEX_OBJECT"
```

`FLAT_OBJECT` chỉnh riêng cho việc đọc được chữ nhỏ trên mặt phẳng: nó đẩy nhiều
hạt sang nhóm bị tách (`percent_dense` 0.005 thay vì 0.01), sinh hạt tới tận
iter 20.000, và kìm tốc độ nở của hạt. `COMPLEX_OBJECT` nới lại ngần ấy cho vật
thể chủ yếu là gờ cạnh, không có chữ.

Một dòng nữa chọn mốc — đây là cách hai tính năng của bản Inria tháng 10/2024
được bật lên từng cái một chứ không bật cả cụm:

```python
MILESTONE = "V2c"   # "V1_5" | "V2a" | "V2b" | "V2c"
```

| mốc | bộ vẽ | cờ mới |
|---|---|---|
| `V1_5` | `dr_aa` | không, và cũng không dùng bảng preset |
| `V2a` | `3dgs_accel` | không |
| `V2b` | `3dgs_accel` | `--antialiasing` |
| `V2c` | `3dgs_accel` | `--antialiasing --optimizer_type sparse_adam` |

`V2a` trông thừa nhưng không thừa. `train.py` truyền
`separate_sh=SPARSE_ADAM_AVAILABLE`, nên chỉ **cài được** bộ vẽ tăng tốc là
đường tính SH đã đổi, chưa bật cờ nào cả. Không có `V2a` thì không có nền trung
tính và cải thiện đến từ đâu cũng không biết. Bù phơi sáng cố tình để ngoài đợt
này — lý do và cách sửa một dòng để bật nó mà không phá `--eval` nằm ở
[`docs/EXPOSURE.md`](docs/EXPOSURE.md).

Ô 5 tính trước bộ ảnh ngốn bao nhiêu RAM và **dừng lại** nếu vượt 10,5 GB, kèm
con số bao nhiêu tấm thì vừa. Ô 7 vá mã nguồn 3DGS cho năm thứ không có đường
dòng lệnh nào tới được: ảnh dạng `uint8`, trần cứng số hạt, phạt hạt bị kéo dài,
log số hạt + VRAM đỉnh mỗi 1000 iter, và bỏ cái `alpha_mask` toàn số 1 mà bản
tháng 10/2024 cất cho từng camera (180 ảnh 3200px là 5,5 GB RAM trả cho một phép
nhân với 1). Nó giữ bản `.bak` và chạy lại được.

Mỗi mốc lưu được chép sang Drive ngay khi xuất hiện, nên Colab có ngắt giữa
chừng thì phần đã xong vẫn còn nguyên.

Ai dùng bản notebook cũ thì lưu ý: biến `LIMIT_VRAM` đã bị bỏ hẳn. Nó nhân đôi
`densify_grad_threshold` và cắt densification sớm 3000 iter — đúng cái làm chữ
nhỏ nhoè thành vệt ở iter 30.000. Nó tồn tại chỉ vì lệnh train cũ thiếu
`--data_device cpu`, để 7,4 GB ảnh nằm chình ình trên VRAM của T4. Giờ cờ đó
luôn được truyền, nên `LIMIT_VRAM` không còn lý do tồn tại.

---

## Trong kho có gì

```
notebooks/
  1_match_images_colab.ipynb              chặng 1 — ghép ảnh bằng GPU
  3_train_gaussian_splatting_colab.ipynb  chặng 3 — huấn luyện
desktop-app/
  quet3d.py                               chặng 2 — app GTK4
  quet3d.desktop                          lối tắt cho GNOME
build-colmap/
  Dockerfile.builder                      nền CUDA 12.2 + Ubuntu 22.04
  01-build-ceres.sh                       Ceres 2.2.0, liên kết tĩnh
  02-build-colmap.sh                      COLMAP 3.13.0, bật CUDA
  03-package.sh                           dò phụ thuộc bằng ldd rồi nhét vào gói
  04-cho-va-dong-goi.sh                   chờ build xong rồi đóng gói
```

Gói COLMAP dựng sẵn nằm ở kho riêng:
**[Ryanhuhut/colmap-cuda-colab](https://github.com/Ryanhuhut/colmap-cuda-colab)**.
Mọi thứ cần để tự dựng lại cũng có đủ trong `build-colmap/` ở đây.

---

## Những cái bẫy nên biết trước

**COLMAP 3.13 đã đổi tên tham số GPU.** Gần như mọi hướng dẫn trên mạng đều viết
theo tên cũ và sẽ báo `unrecognised option`:

| Tên cũ (hỏng) | Tên đúng cho 3.13 |
|---|---|
| `--SiftExtraction.use_gpu` | `--FeatureExtraction.use_gpu` |
| `--SiftMatching.use_gpu` | `--FeatureMatching.use_gpu` |

Cả hai mặc định đã là `1`. Khi nghi ngờ, hỏi thẳng file `colmap` thay vì tin
hướng dẫn trên mạng: `colmap feature_extractor -h | grep use_gpu`.

**Gói dựng sẵn chỉ chạy trên sm_75 — tức Tesla T4.** Gặp L4 hay A100 sẽ báo lỗi
kiến trúc CUDA. Nếu bạn dùng Colab trả phí thì build lại với
`CUDA_ARCH="75;80;89"` trong `02-build-colmap.sh`.

**Đừng build trên Ubuntu 24.04.** Chương trình biên dịch trên hệ cũ chạy được
trên hệ mới, ngược lại thì không. Build trên 22.04 cho ra thứ chạy được cả hai —
đã thử thật trên container 22.04 và 24.04 sạch.

**Ceres trong Ubuntu 22.04 quá cũ.** COLMAP 3.13 dùng `ceres::Manifold`, thứ chỉ
có từ Ceres 2.1. Ubuntu 22.04 chỉ có 2.0.0, nên Ceres được build từ nguồn và
liên kết tĩnh — nhờ vậy Colab khỏi phải cài Ceres.

**Phải nhét `libgomp` vào gói.** Image Ubuntu trần không có sẵn nó. Bỏ sót thì
gói chạy ngon trên máy build rồi chết trên Colab với
`error while loading shared libraries: libgomp.so.1`.

**Tuyệt đối không nhét `libcuda.so.1` vào gói.** Đó là trình điều khiển card,
bắt buộc phải lấy từ chính máy có GPU.

**Máy bật SELinux thì lệnh mount Docker phải có hậu tố `:z`.** Fedora, RHEL,
CentOS. Thiếu là `Permission denied`.

---

## Cần những gì

**Chặng 1 và 3:** một tài khoản Google. Colab bậc miễn phí là đủ.

**Chặng 2:** máy Linux có Docker, chừng 3GB đĩa trống, GTK4 và libadwaita.
Không cần GPU — app không hề đụng tới card nào.

**Nếu muốn tự build lại COLMAP:** Docker và chừng 20GB đĩa trống. Vẫn không cần
GPU; biên dịch mã CUDA chỉ cần `nvcc`, thứ nằm sẵn trong image nền. Build mất
khoảng 30 phút với 4 luồng.

---

## Gói COLMAP

Chặng 1 và chặng 2 dùng chung **đúng một bản COLMAP**, nhờ vậy database ghi trên
Colab được đọc bởi đúng phiên bản ấy trên máy bạn. Trộn hai phiên bản khác nhau
là cách nhanh nhất để nhận lỗi `SQL logic error` mà không hiểu vì sao.

Bản mới nhất, luôn tự trỏ đúng bản cuối:

```
https://github.com/Ryanhuhut/colmap-cuda-colab/releases/latest/download/colmap-3.13.0-cuda12.2-ubuntu2204-sm75.tar.gz
```

Bản ghim theo phiên bản, dùng khi cần lấy lại đúng bản này về sau:

```
https://github.com/Ryanhuhut/colmap-cuda-colab/releases/download/v3.13.0-cuda12.2-sm75/colmap-3.13.0-cuda12.2-ubuntu2204-sm75.tar.gz
```

Nặng 46MB (47.654.853 byte).

```
SHA256: 6c72e8535a780198ce5e56af01d9aac4447de32051cffa9a773733ebcd9ac317
```

Kho chứa, kịch bản build và ghi chú phát hành:
**[Ryanhuhut/colmap-cuda-colab](https://github.com/Ryanhuhut/colmap-cuda-colab)**

---

## Mọi thứ được dùng ở đây

Không phần khó nào trong này là của tôi. Kho này chỉ là đường ống: nó nối các
công cụ có sẵn lại, biên dịch một trong số đó cho đúng cách, và chia việc ra hợp lý.

### Hai thứ làm việc thật sự

| Dự án | Vai trò | Giấy phép |
|---|---|---|
| [COLMAP](https://github.com/colmap/colmap) | Dựng cấu trúc từ chuyển động — tìm ra mỗi ảnh chụp từ đâu | BSD |
| [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Bộ dựng hình, của Inria và MPII | Nghiên cứu phi thương mại |

### Nằm sẵn trong gói COLMAP

| Thư viện | Phiên bản | Vai trò | Giấy phép |
|---|---|---|---|
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | 12.2.140 | Tính toán trên GPU | NVIDIA EULA |
| [Ceres Solver](https://github.com/ceres-solver/ceres-solver) | 2.2.0 | Tối ưu phi tuyến | BSD |
| [Eigen](https://eigen.tuxfamily.org) | 3.4.0 | Đại số tuyến tính | MPL2 |
| [Boost](https://www.boost.org) | 1.74.0 | Đọc tham số dòng lệnh, đồ thị | Boost |
| [PoseLib](https://github.com/PoseLib/PoseLib) | `f119951` | Giải bài toán vị trí camera | BSD |
| [faiss](https://github.com/facebookresearch/faiss) | `36b7735` | Tìm hàng xóm gần nhất | MIT |
| [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) | 5.10.1 | Giải hệ ma trận thưa | LGPL/GPL |
| [OpenBLAS](https://www.openblas.net) | 0.3.20 | Đại số tuyến tính cơ bản | BSD |
| [LAPACK](https://www.netlib.org/lapack/) | 3.10.0 | Đại số tuyến tính bậc cao | BSD |
| [METIS](https://github.com/KarypisLab/METIS) | 5.1.0 | Chia nhỏ đồ thị | Apache 2.0 |
| [CGAL](https://www.cgal.org) | 5.4 | Hình học tính toán | GPL/LGPL |
| [FreeImage](https://freeimage.sourceforge.io) | 3.18.0 | Đọc ghi ảnh | FIPL/GPL |
| [glog](https://github.com/google/glog) | 0.4.0 | Ghi nhật ký | BSD |
| [gflags](https://github.com/gflags/gflags) | 2.2.2 | Xử lý cờ dòng lệnh | BSD |
| [SQLite](https://www.sqlite.org) | 3.37.2 | Database đặc trưng và cặp ảnh | Phạm vi công cộng |

Kèm theo là các thư viện ảnh mà FreeImage kéo theo — libjpeg, libpng, libtiff,
libwebp, libraw, OpenEXR, JPEG-XR — cùng zlib, OpenSSL, libcurl và libgomp.
Tổng cộng **79 thư viện dùng chung**; danh sách đầy đủ nằm trong `BUILD_INFO.txt`
bên trong gói.

### App máy tính

| Dự án | Vai trò | Giấy phép |
|---|---|---|
| [GTK4](https://www.gtk.org) | Bộ công cụ giao diện | LGPL |
| [libadwaita](https://gitlab.gnome.org/GNOME/libadwaita) | Kiểu dáng chuẩn GNOME | LGPL |
| [PyGObject](https://pygobject.gnome.org) | Cầu nối Python cho cả hai thứ trên | LGPL |
| [Docker](https://www.docker.com) | Chạy COLMAP mà không cài nó lên máy | Apache 2.0 |

### Xem kết quả

| Dự án | Vai trò |
|---|---|
| [SuperSplat](https://superspl.at/editor) | Trình xem và chỉnh sửa file `.ply` ngay trên trình duyệt |

### Chạy ở đâu

[Google Colab](https://colab.research.google.com) bậc miễn phí — một card Tesla
T4 với 15GB VRAM, và hai nhân CPU.

---

## Giấy phép

Mã trong kho này dùng giấy phép MIT. Các công cụ mà nó điều khiển giữ giấy phép
riêng: COLMAP dùng BSD, còn Gaussian Splatting chỉ miễn phí cho nghiên cứu phi
thương mại — hãy đọc điều khoản của họ trước khi dùng vào việc kinh doanh.
