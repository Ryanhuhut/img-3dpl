# Chế độ quét phòng — đang khoá

Ghi lại để lần sau mở lại không phải điều tra từ đầu.

**Trạng thái:** khoá từ 2026-08-14. Chọn `PRESET = "ENTIRE_ROOM"` ở ô 6 của
notebook 3 sẽ báo lỗi ngay và dừng.

**Vì sao khoá chứ không xoá:** phần chặng 3 đã viết xong và đã chạy thử. Xoá đi
là vứt công, mà để mở thì người dùng chạy hết ba tới năm tiếng rồi nhận về một
mô hình không dùng được. Một tính năng nửa vời tệ hơn không có tính năng.

---

## Cái gì đã xong

Toàn bộ ở **chặng 3**, còn nguyên trong `notebooks/3_train_gaussian_splatting_colab.ipynb`,
không dòng nào bị xoá:

| chỗ | nội dung |
|---|---|
| `PRESETS["ENTIRE_ROOM"]` | bộ tham số đầy đủ: `densify_grad_threshold` 0.0004, `sh_degree` 2, trần 2.000.000 hạt, `MIN_OPACITY` 0.01 |
| `HOP_LE` / `LOAI_QUET` | nhánh `"ROOM"` và ràng buộc `ROOM -> ENTIRE_ROOM` |
| `get_training_config()` | kiểm tra `ROOM` + masking bật = sai, raise ngay |
| ô 5 — ngân sách | `sh_degree 2` cho 38 float/hạt thay vì 59, tính đúng cho phòng |
| ô 10 — `compress_ply()` | lọc `opacity < 0.02` và cắt hạt vượt phân vị scale 99,5% |

Đã chạy thử ở máy: 400 ảnh @ 1600px ra 2,30 GB RAM và VRAM đỉnh ~3,9 GB, đều
dưới trần. Bộ vá áp đúng `sh_degree 2` và trần 2 triệu hạt.

---

## Ba việc còn thiếu

### 1. Chặng 1 dùng sai công cụ ghép ảnh

`notebooks/1_match_images_colab.ipynb` chạy `exhaustive_matcher` — so mọi ảnh
với mọi ảnh. Với **vật thể** thì đó là lựa chọn đúng và là lý do repo này tồn
tại: đi vòng quanh một vật thể thì tấm cuối thật sự chồng lấn tấm đầu, và
`sequential_matcher` bỏ sót đúng chỗ khép vòng đó nên mô hình bị trôi lệch.

Với **phòng** thì lập luận trên sụp đổ. Đi bộ trong phòng, ảnh chụp góc bắc và
ảnh chụp góc nam **không nhìn thấy gì chung**. Cặp đó không thể có kết quả — mà
COLMAP vẫn phải dò đặc trưng, so descriptor, chạy RANSAC, rồi mới kết luận là
rỗng. Tiền vẫn mất, hàng không có.

Số cặp tăng theo bình phương số ảnh, `n(n-1)/2`:

| số ảnh | số cặp | ghi chú |
|---|---|---|
| 320 | 51.040 | số đo thật của repo: 28 phút trên T4 |
| 400 | 79.800 | |
| 500 | **124.750** | gấp 2,4 lần mức đã đo |

Phần lớn con số 124.750 ấy là cặp rỗng. Đó là định nghĩa của sai công cụ.

**Hướng sửa (chưa thử):** `sequential_matcher` cộng phát hiện khép vòng bằng
vocabulary tree. Đi bộ trong phòng thì ảnh liền kề nhau về thời gian cũng liền
kề nhau về không gian, nên `sequential` đúng với cấu trúc bài toán; vocab tree
lo phần bắt lại chỗ quay về. Đánh đổi: phải tải file vocab tree về, và chất
lượng khép vòng kém hơn exhaustive.

**Chưa kiểm chứng:** COLMAP 3.13 đã đổi tên `SiftExtraction.*` thành
`FeatureExtraction.*`. Rất có thể họ `SequentialMatching.*` cũng đã đổi. Phải
chạy `colmap sequential_matcher -h` mà đọc, **không được đoán tên tham số.**

### 2. Chặng 2 nghẹn, và chưa ai đo thật

Số đo có thật: mapper trên Intel i3-1005G1 hai nhân mất **105 phút cho 320
ảnh**. SfM tăng dần là tuần tự và siêu tuyến tính theo số ảnh, nên 500 ảnh
**ước lượng** rơi vào 3–5 tiếng. Đây là ước lượng, không phải số đo — chưa ai
chạy thử.

Việc (1) ở trên ăn thẳng vào việc này: ít cặp hơn thì mapper nhẹ đi nhiều.

Ba hướng khác, chưa hướng nào thử:

- **Chia nhỏ:** quét phòng thành 2–3 vùng chồng lấn, dựng riêng từng vùng, ghép
  bằng `model_merger` của COLMAP.
- **Đưa mapper lên Colab.** Ngược với triết lý hiện tại của repo, nhưng đáng đo
  lại: repo giả định "laptop nhiều nhân hơn Colab", mà i3-1005G1 có đúng **2
  nhân** — bằng Colab free. Giả định nền tảng ấy không đúng với chính máy của
  tác giả.
- **Chấp nhận số ảnh thấp hơn.** 300 ảnh thay vì 500. Đổi độ phủ lấy thời gian.

### 3. Dọn hạt ma cho phòng mới có mỗi dòng nhắc

Ô 10 in ra hai gạch đầu dòng nhắc việc phải làm tay. Đó là chú thích, không
phải tính năng.

Có sẵn (bước 2 trong danh sách dưới): `compress_ply()` lọc `opacity < 0.02` và
cắt hạt vượt phân vị scale 99,5%.

Thứ tự đúng, rẻ trước:

1. **Gốc rễ, lúc chụp:** khoá phơi sáng và cân bằng trắng bằng tay. Phần lớn
   hạt ma sinh ra vì 3DGS phải bịa hình học để giải thích cùng một điểm sáng
   khác nhau giữa các ảnh. Điện thoại để auto là nguồn hạt ma lớn nhất khi quét
   phòng. Không code nào sửa được sau khi đã chụp xong.
2. **Lọc opacity + scale.** ✅ đã có.
3. **Đếm tầm nhìn.** ❌ chưa có. Với mỗi hạt, đếm số camera frustum chứa nó
   (đọc pose từ `sparse/0/images.bin`). Thấy bởi dưới 3 camera thì xoá. An toàn
   với tường, hiệu quả với hạt ma.
4. **Statistical Outlier Removal.** ❌ chưa có. k=16 láng giềng,
   `std_ratio=2.0`. Open3D, hoặc `3dgsconverter` đã tích hợp sẵn bản chạy GPU.
5. **Crop tay trong SuperSplat.** Hai phút, và hiệu quả nhất trong cả năm cách.

**[ĐÃ CHỐT] Không dùng heuristic "xoá hạt cách camera quá X mét."** Trong
phòng, tường luôn xa camera. Áp luật đó là thủng tường. Ghi ra đây vì đó là
cách đầu tiên ai cũng nghĩ tới.

---

## Mở lại thế nào

1. Làm xong ba việc trên.
2. Đặt `CHAN_ROOM = False` trong ô 6 của notebook 3.
3. Trả `"ENTIRE_ROOM"` lại vào phần chú thích `PRESET` mà người dùng nhìn thấy,
   và trả cột `ENTIRE_ROOM` lại vào bảng preset ở ô markdown số 5.
4. Cập nhật `README.md` và `README.vi.md` — hiện đang nói thẳng là quét phòng
   làm dở.

Cổng chặn nằm gọn trong một khối có đánh dấu ở ô 6, gồm `CHAN_ROOM`,
`THIEU_GI_CHO_ROOM`, và ba dòng kiểm tra trong `get_training_config()`. Gỡ khối
đó ra là xong, không phải sửa gì khác.

---

## Ràng buộc vẫn còn hiệu lực

Thứ tự thử ở `docs/V2_PLAN.md` mục A.2.5 — **không gộp mốc**. Quét phòng nằm
sau toàn bộ đường vật thể. Tính tới lúc viết ghi chú này, **Mốc A còn chưa
chạy**, nên mọi việc trong tài liệu này xếp sau Mốc A, B và C.
