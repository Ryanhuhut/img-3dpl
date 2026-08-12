# img-3dpl

***img**age **-** to **3d** **p**ipe**l**ine — quy trình biến ảnh thành 3D*

Biến một thư mục ảnh chụp thường thành mô hình Gaussian Splatting, trên máy tính
không có card đồ hoạ nào chạy được.

[English version → README.md](README.md)

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
   ảnh    ──►  [1] GPU Colab      ──►  database.db
                   ghép các cặp

database.db ──►  [2] app máy tính ──►  images/ + sparse/0/
   + ảnh            dựng camera

  sparse/0 ──►  [3] GPU Colab     ──►  mô hình .ply
                   huấn luyện
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

### Chặng 1 — ghép ảnh, trên Colab

Mở [`notebooks/1_match_images_colab.ipynb`](notebooks/1_match_images_colab.ipynb)
bằng Colab, chọn `Runtime` → `Change runtime type` → **T4 GPU**, sửa ô cấu hình,
rồi chạy hết. Kết quả là file `database.db` nằm trên Drive của bạn.

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

Tải `database.db` từ Drive về, rồi chạy app. Nó chỉ cần Docker, không cần gì
khác — COLMAP chạy bên trong container nên máy bạn không bị cài thêm thứ gì.

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

Cần GTK4 và libadwaita — hai thứ có sẵn trên mọi bản GNOME hiện hành
(`python3-gobject gtk4 libadwaita`).

### Chặng 3 — huấn luyện, trên Colab

Đưa thư mục kết quả lên Drive, mở
[`notebooks/3_train_gaussian_splatting_colab.ipynb`](notebooks/3_train_gaussian_splatting_colab.ipynb),
sửa ô cấu hình, chạy hết. Mỗi mốc lưu được chép sang Drive ngay khi xuất hiện,
nên Colab có ngắt giữa chừng thì phần đã xong vẫn còn nguyên.

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
