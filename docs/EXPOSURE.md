# Bù phơi sáng — hoãn sang đợt sau

Ghi lại để lần sau mở lại không phải điều tra từ đầu.

**Trạng thái:** hoãn từ 2026-08-16. Đợt cập nhật lên bản Inria tháng 10/2024 chỉ
lấy `--antialiasing` và `--optimizer_type sparse_adam`. Bốn cờ
`--exposure_lr_*` **không** được thêm vào notebook 3.

**Vì sao hoãn:** cách dùng mà Inria ghi trong README kéo theo `--train_test_exp`,
và cờ đó phá thước đo mà cả chặng 3 đang dựa vào. Gộp bù phơi sáng chung với
chống răng cưa trong một đợt thì cũng không biết cái nào ăn tiền. Nhưng có một
đường ra không phải trả giá đó — mục "Đường ra" ở dưới.

---

## Vấn đề có thật, không phải chuyện lý thuyết

Ảnh chụp bằng điện thoại để phơi sáng tự động. Mỗi góc chụp máy tự chọn một độ
sáng khác nhau, nên cùng một điểm trên vật thể có hai giá trị sáng khác nhau ở
hai tấm ảnh. 3DGS khớp theo cường độ sáng, nên cách duy nhất optimizer dung hoà
được là bịa ra hình học để giải thích sự chênh lệch ấy — đúng cơ chế sinh hạt ma
đã ghi ở [`ROOM_MODE.md`](ROOM_MODE.md) mục 3.

Bản tháng 10/2024 học một phép biến đổi affine 3×4 riêng cho **từng ảnh**, đúng
như [Hierarchical 3DGS](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/).

Cách chữa rẻ nhất vẫn nằm ở lúc chụp: khoá phơi sáng và cân bằng trắng bằng tay.
Không code nào sửa được sau khi đã bấm máy. Bù phơi sáng chỉ là lưới đỡ.

---

## Bốn cờ, và cái bẫy

Tên cờ lấy từ `arguments/__init__.py`, lớp `OptimizationParams`:

| cờ | mặc định | Inria khuyến nghị |
|---|---|---|
| `--exposure_lr_init` | 0.01 | 0.001 |
| `--exposure_lr_final` | 0.001 | 0.0001 |
| `--exposure_lr_delay_steps` | 0 | 5000 |
| `--exposure_lr_delay_mult` | 0.0 | 0.001 |

**Bốn cờ này một mình không làm gì cả.** `train.py` gọi render như sau:

```python
render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=SPARSE_ADAM_AVAILABLE)
```

và `gaussian_renderer/__init__.py` chỉ áp phép biến đổi khi cờ đó bật:

```python
if use_trained_exp:
    exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
    rendered_image = torch.matmul(...) + exposure[:3, 3, None, None]
```

Không bật `--train_test_exp` thì `_exposure` không bao giờ vào đồ thị tính loss,
không có gradient, và `gaussians.exposure_optimizer.step()` chạy trên số không.
Lệnh mẫu ở README của Inria có `--train_test_exp` ở cuối — dễ đọc sót.

---

## Vì sao `--train_test_exp` phá thước đo

Cờ đó không chỉ bật bù phơi sáng. Nó đổi luôn nghĩa của `--eval`:

| chỗ | làm gì |
|---|---|
| `scene/dataset_readers.py` | `train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]` — **toàn bộ ảnh đối chứng được đưa vào cả tập train** |
| `scene/cameras.py` | ảnh đối chứng bị che nửa: bản trong tập train che nửa phải, bản trong tập test che nửa trái |
| `render.py` | chỉ cắt nửa phải ra để đo |

Nghĩa là PSNR/SSIM ở ô 9 sẽ đo trên **nửa ảnh**, của một mô hình đã được nhìn
thấy nửa còn lại. Bảng bốn mốc `V1_5 / V2a / V2b / V2c` mất giá trị so sánh ngay
lập tức — số cũ và số mới không cùng đơn vị đo nữa.

Đó là toàn bộ lý do hoãn. Không phải vì tính năng dở.

---

## Đường ra: sửa một dòng, thước đo còn nguyên

`use_trained_exp` là **tham số của hàm `render`**, không phải cùng một thứ với
`train_test_exp`. `train.py` chỉ đang tiện tay lấy cờ này làm giá trị cho cờ kia.
Tách hai thứ đó ra là xong:

```python
# trong train.py, vòng lặp train
render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                    use_trained_exp=True,          # <- thay dataset.train_test_exp
                    separate_sh=SPARSE_ADAM_AVAILABLE)
```

Kết quả: bù phơi sáng chạy thật trên ảnh train, còn `--train_test_exp` vẫn tắt
nên `--eval` giữ nguyên nghĩa cũ và bảng bốn mốc vẫn so được.

Việc này thuộc dạng "không có đường CLI nào tới được", nên nó sẽ là **VÁ 6** ở
ô 7 của notebook 3, cùng chỗ với năm bản vá đang có.

### Bốn thứ phải kiểm trước khi tin

1. **`render.py` sẽ KHÔNG áp bù phơi sáng.** Nó truyền
   `use_trained_exp=dataset.train_test_exp`, mà cờ đó vẫn tắt. Ảnh render ở ô 9
   là ảnh chưa bù. Điều này **có lợi cho phép đo** — nó giữ ô 9 đo đúng thứ ô 9
   vẫn đo — nhưng phải biết rõ chứ đừng ngạc nhiên. Đừng "sửa cho đồng bộ" bằng
   cách vá luôn `render.py`: làm thế là quay lại đúng chỗ vừa tránh.

2. **Ảnh đối chứng không có mục exposure.** `exposure_mapping` dựng từ
   `scene_info.train_cameras`, nên gọi `get_exposure_from_name` với tên một ảnh
   test sẽ là `KeyError`. Với `--train_test_exp` tắt thì `render.py` không gọi
   tới đó, nên không sao — nhưng đây là dây mìn nếu sau này ai đó bật `use_trained_exp`
   trong `render.py`.

3. **File `.ply` không đổi.** Bù phơi sáng ghi ra `exposure.json` ở thư mục gốc
   của model (`scene/__init__.py`, hàm `save`), không ghi vào `.ply`. SuperSplat
   mở `.ply` sẽ thấy đúng độ sáng đã hội tụ. Đây là điều **đang mong muốn**: các
   hạt thôi phải hấp thụ chênh lệch phơi sáng, còn người xem thì không phải bù gì.

4. **Đo bằng một mốc riêng.** `V2d = V2c + bù phơi sáng`. Đừng gộp vào V2c đang
   có, không thì lại rơi vào đúng cái bẫy đợt này vừa tránh.

---

## Làm lúc nào

Sau khi bảng bốn mốc `V1_5 / V2a / V2b / V2c` đã có số thật. Trước đó thì chưa
có nền để so, mà đo bù phơi sáng trên một cái nền chưa biết là vô nghĩa.

Nếu ảnh đầu vào chụp bằng chân máy với phơi sáng khoá tay thì bỏ qua hẳn mục
này — không có gì để bù.
