# Nhận diện biển báo giao thông hình tròn

Chương trình giữa kỳ môn Xử lý ảnh số: đọc từng frame, tìm biển báo tròn ở khu vực giữa/phải ảnh, theo dõi qua thời gian và vẽ một bounding box quanh mỗi biển được chấp nhận. Cùng một cấu hình xử lý cả `video1.mp4` và `video2.mp4`.

Mã nguồn chỉ import **OpenCV, NumPy và math**, không dùng mô hình huấn luyện sẵn, không đặt quy tắc theo số frame, thời điểm hay tọa độ từng biển cụ thể. Chương trình xác định vị trí biển, chưa phân loại tên/ý nghĩa chính xác của biển.

Pipeline có lớp `SignDetector`, phần chuẩn hóa sáng và tracker hỗ trợ optical flow như mô tả dưới đây. Đối chiếu mã băm trong manifest cục bộ để biết video có khớp phiên bản code đang dùng hay không; kết quả của bản trước không tự chứng minh bản mới đúng.

## 1. Cài đặt và chạy

Đặt hai video nguồn cùng thư mục với `detect_circular_signs.py`. Chuẩn bị Python 3.11 và chạy PowerShell tại thư mục dự án:

```powershell
python -m pip install opencv-python numpy
New-Item -ItemType Directory -Path output -Force
python detect_circular_signs.py
```

Có thể bấm Run Python File trong trình soạn thảo sau khi đã tạo thư mục `output`. Code tìm video theo vị trí file Python, không theo thư mục đang mở trong terminal.

Cấu hình đầu file:

```python
STUDENT_ID = "TEST"  # thay bằng MSSV; nhiều MSSV nối bằng dấu _
INPUT_VIDEOS = ("video1.mp4", "video2.mp4")
OUTPUT_FOLDER = "output"
MAX_SECONDS = None  # None: toàn video; 20: thử 20 giây đầu
```

- Muốn chỉ chạy video 1: đặt `INPUT_VIDEOS = ("video1.mp4",)`.
- `OUTPUT_FOLDER` phải tồn tại. Có thể đặt `""` để lưu kết quả cùng thư mục với code. Chương trình dùng ba thư viện đúng giới hạn đề, nên không import thêm thư viện để tự tạo thư mục.
- Không dùng argparse; không cần tham số dòng lệnh.
- Mỗi video nguồn tạo đúng một file kết quả trong một lần chạy. Chạy lại cùng cấu hình sẽ ghi đè file cùng tên.
- Chỉ in thông báo khi hoàn thành mỗi video hoặc khi có lỗi; không in tiến độ từng nhóm frame.

Với cấu hình mặc định:

```text
output/video1_TEST.mp4
output/video2_TEST.mp4
```

Nếu `MAX_SECONDS = 20`, tên có hậu tố `_preview_20s`. Các bản thử cũ trong máy không tự bị xóa; chương trình hiện tại không tự sinh thêm bản shape/test.

## 2. Luồng xử lý

```text
Đọc frame gốc
  → Cắt ROI có thêm lề quanh vùng tâm được phép
  → Blur, HSV và hỗ trợ độ sáng bằng CLAHE trên L của LAB
  → Hai màu đỏ/xanh, hai mức bão hòa và mặt nạ hợp đỏ–xanh
  → CLOSE / CLOSE+OPEN → contour
  → Cùng một bộ kiểm tra hình dạng, vị trí và kích thước
  → Chuẩn hóa elip về vùng tròn để đo màu, vành và ký hiệu
  → Cùng một công thức điểm → loại box trùng
  → Tracker: xác nhận, ghép, làm mượt, hỗ trợ mất dấu ngắn
  → Vẽ box và MSSV lên frame gốc → ghi MP4
```

Detector chạy trên **mọi frame**. Optical flow chỉ được gọi cho track mà detector không ghép được trong frame hiện tại.

Các nhánh cũ `red_faded`, `red_recovered`, `red_lighting`, Hough và ghép hai vùng đỏ không còn các bộ điều kiện/điểm riêng. Ứng viên màu yếu, màu mạnh và mặt nạ hợp đều đi qua cùng một đường đánh giá. Việc tạo ứng viên và việc kết luận đó là biển được tách rõ.

## 3. Các bước phát hiện

### 3.1. ROI và độ phân giải

`CENTER_ROI = (0.12, 0.18, 1.00, 0.48)` giới hạn **tâm** ứng viên:

```text
0.12W ≤ cx ≤ W
0.18H ≤ cy ≤ 0.48H
```

Chương trình cắt ROI có lề `0.10W` xung quanh để tránh cắt mất vành của biển có tâm nằm gần mép vùng này. Độ phân giải nguồn được giữ khi phát hiện; không thu nhỏ toàn frame làm mất chi tiết biển xa.

Đây là giới hạn theo ảnh nhằm bỏ vùng quá cao, vùng thấp và sát mép trái. Nó chưa nhận biết hình học lòng đường. Video có góc camera khác có thể cần chỉnh **một ROI chung**, rồi áp dụng lại cho tất cả video cần đánh giá.

Bán kính trục lớn của elip cần nằm trong:

```text
0.0042W ≤ r ≤ 0.10W
```

Ở chiều rộng 1920 pixel, khoảng này là 8.064–192 pixel. Kernel làm mờ, morphology, khoảng ghép tối thiểu và nét vẽ cũng được quy đổi theo độ phân giải. Kích thước ảnh chuẩn hóa và cửa sổ LK là kích thước lấy mẫu nội bộ, không phải kích thước biển trong video.

### 3.2. Màu và độ sáng

H của HSV OpenCV nằm trong 0–179; S và V nằm trong 0–255.

| Màu | H | S tối thiểu | V tối thiểu |
| --- | --- | --- | --- |
| Đỏ | 0–15 hoặc 140–179 | 35 | 45 |
| Xanh | 94–132 | 65 | 45 |

Mặt nạ bão hòa cao dùng cùng H/V và tăng ngưỡng S lên `1.8` lần: đỏ 63, xanh 117. Nó giúp tách một biển khỏi vùng nền mà mặt nạ màu rộng đã nối vào. Mặt nạ hợp đỏ–xanh hỗ trợ lấy toàn mặt biển có viền đỏ, nền xanh.

Nếu nền vẫn dính vào mặt nạ bão hòa cao, `split_color_components()` lấy các thành phần liên thông và dùng Otsu trên S trong từng thành phần để tạo thêm mặt nạ phần bão hòa cao hơn. Ngưỡng tách được tính từ ảnh, dùng chung cho đỏ/xanh và mọi video. Mặt nạ gốc vẫn được giữ để tránh mất biển bạc màu. Tất cả ứng viên bổ sung đều phải qua cùng kiểm tra hình dạng, màu, cấu trúc và điểm; không tự tạo track chỉ vì vượt ngưỡng Otsu.

Khi bật `ENABLE_LIGHT_NORMALIZATION`:

1. Chuyển ROI sang LAB.
2. CLAHE trên L với `clipLimit = 2.0`, lưới 8×8.
3. Trộn 60% L gốc và 40% L đã xử lý.
4. Đổi về HSV, chỉ lấy V lớn hơn giữa ảnh này và ảnh làm mờ gốc. Giữ H/S gốc cho việc lọc màu.

Cách này hỗ trợ pixel bị tối mà không tự đổi một sắc màu khác thành đỏ. Nó không khắc phục hoàn toàn màu bạc, lệch cân bằng trắng, cháy sáng hoặc nhòe. Frame dùng xuất video vẫn là frame gốc; ảnh chuẩn hóa chỉ dùng phân tích.

### 3.3. Tạo contour, giữ cả vành mỏng

Mỗi mặt nạ tạo hai biến thể:

- CLOSE để nối khe nhỏ nhưng giữ viền mỏng.
- CLOSE rồi OPEN để gỡ các cầu nối nhỏ giữa biển và nền.

Với 1920 pixel, kernel là elip 3×3. Cả hai biến thể dùng cùng validator và không có bộ điểm riêng. Trước khi khớp elip, loại contour quá nhỏ, quá lớn hoặc nằm ngoài vùng tìm kiếm để giảm tính toán thừa.

Chương trình dùng contour ngoài (`RETR_EXTERNAL`). Box trùng từ các biến thể sẽ được loại sau khi chấm điểm.

### 3.4. Điều kiện hình dạng bắt buộc

`contour_geometry()`:

1. Lấy convex hull để kiểm tra dạng chữ nhật và tính độ gọn.
2. Loại hull xấp xỉ có tối đa 4 đỉnh và chiếm hơn 87% hình chữ nhật xoay bao ngoài.
3. Khớp elip lên contour thật bằng `fitEllipse()` và kiểm tra sai lệch tuyệt đối ở phân vị 90% như trước. Nếu đạt ngưỡng 0.12 thì giữ cách xử lý gốc.
4. Yêu cầu tỷ lệ trục nhỏ/trục lớn ≥ 0.60.
5. Nếu contour không đạt do có vết lõm, thử khớp elip lên hull. Đưa các điểm contour thật về hệ tọa độ elip và tính:

```text
residual = sqrt(u² + v²) − 1
E_boundary = phân vị MIN_BOUNDARY_SUPPORT của abs(residual)
E_outward = phân vị 90% của max(residual, 0)
```

`u`, `v` được tính trên contour thật, đã trừ tâm, xoay theo elip và chia cho hai bán trục. Khi phục hồi cần cả `E_boundary ≤ 0.12` và `E_outward ≤ 0.12`. `MIN_BOUNDARY_SUPPORT = 0.60` nghĩa là ít nhất khoảng 60% điểm biên thật phải sát elip. Vùng lõm trong tâm còn phải có ít nhất 70% pixel được xác định là ký hiệu sáng (`MIN_LIGHT_NOTCH_RATIO`), và chiếm tối thiểu 3% vùng tâm (`MIN_LIGHT_NOTCH_AREA_RATIO`). Điều kiện diện tích vết lõm thuộc kiểm tra hình học, không phụ thuộc tham số chấm điểm ký hiệu xanh. Nhờ đó chỉ cho phép vết lõm có bằng chứng ảnh, không tự nới hình dạng cho mọi mảng màu.

Không chỉ đo sai lệch trên hull: thao tác đó có thể làm một mảng cờ hoặc nền gồ ghề trông tròn hơn thực tế. Kiểm tra hình dạng này cho phép vết lõm, nhưng các điều kiện màu quanh vành và ký hiệu bên trong vẫn phải đạt; nới kiểm tra biên có thể thêm ứng viên sai nên cần kiểm tra hồi quy. Chương trình cũng không yêu cầu mọi biển nghiêng phải đạt circularity của hình tròn chính diện.

Điểm hình dạng:

```text
C_hull = 4π × diện_tích_hull / chu_vi_hull²
E = sai lệch phân vị 90% ở nhánh gốc, hoặc E_boundary khi phục hồi có bằng chứng ký hiệu
shape = 0.50 × cap(C_hull / 0.90) + 0.50 × cap(1 − E)
```

`cap(x) = min(1, max(0, x))`. Không bắt buộc vùng màu phải lấp đầy toàn elip vì biển có thể chỉ còn một phần vành; bằng chứng màu quanh cả vành và cấu trúc bên trong vẫn phải đạt bước sau.

### 3.5. Chuẩn hóa elip và đo bằng chứng

`ellipse_patch()` dùng remap đưa elip về hệ tròn 64×64, gồm cả lề tới 1.4 lần bán kính. Nhờ đó, biển nghiêng và biển chính diện được đo trên cùng vùng chuẩn hóa. Đây là chuẩn hóa hình học 2D, không khôi phục chi tiết đã mất hoặc mọi méo phối cảnh.

`appearance_features()` đo ba nhóm bằng chứng:

| Thành phần | Cách đo |
| --- | --- |
| `support` | 72 góc, bốn bán kính 0.72/0.84/0.96/1.04, chia 12 phần vành. Một phần có màu nếu ít nhất 20% mẫu thuộc màu đó. Support là tỷ lệ phần đạt. |
| `pattern` | Mức phù hợp của nền sáng có chi tiết tối, dải sáng giữa hai vùng đỏ, nền xanh có phần sáng hoặc có vành đỏ. |
| `isolation` | Giảm điểm khi cùng màu của ứng viên tiếp tục lan nhiều ra vùng 1.12–1.40 bán kính ở bên ngoài. |

Độ sáng phân biệt vùng sáng/tối dựa trên `V90` trong tâm: sáng có V ≥ `max(45, 0.60V90)`; tối có V < `0.65V90`. Với biển đỏ, pixel sáng cần S < `min(75, 0.75 × S_đỏ)`, trong đó `S_đỏ` là phân vị 90% S của pixel đỏ trên vành; nếu không có pixel đỏ thì dùng 75. Tham chiếu từ vành giúp tránh nhầm nền trắng bị ám đỏ thành phần màu đỏ dùng để đặt ngưỡng, đồng thời một mảng đỏ nhạt đồng nhất vẫn không tự được tính là nền trắng. Riêng ký hiệu trên nền xanh cần S < `SYMBOL_SATURATION_RATIO × S_nền`, với `S_nền` là trung vị độ bão hòa các pixel xanh trong tâm và tỷ lệ mặc định 0.75. Như vậy ký hiệu có thể bị lem xanh, nhưng phải ít bão hòa hơn nền xanh ít nhất khoảng 25%; một mảng xanh nhạt đồng nhất không tự được tính vừa là nền vừa là ký hiệu. Không chặn thêm ở S=130 vì mũi tên thật bị lem xanh có thể vượt mức này.

Ngoài việc có màu quanh vành, `boundary_features()` yêu cầu màu phải kết thúc gần biên elip. Với từng màu, loại pixel có S thấp hơn 75% trung vị S của vành để giảm ảnh hưởng nền cây yếu màu. So sánh cùng 72 góc tại bốn bán kính trong và ba bán kính ngoài (1.12/1.24/1.36). Một sector có biên khi tỷ lệ màu trong ≥20% và cao hơn ngoài ít nhất 20 điểm phần trăm. Cần đạt `MIN_RADIAL_SUPPORT = 0.65`, tương ứng ít nhất 8/12 sector. Đây là điều kiện bắt buộc; điểm hình dạng cao không bù được việc thiếu biên màu. Điểm `isolation = 1 − cap(tỷ_lệ_màu_ngoài / 0.40)` cũng dùng chính mặt nạ màu đã lọc này, tránh trừ điểm biển chỉ vì nền cây phía sau có màu xanh yếu hơn.

Các mức tham chiếu của `pattern`:

- Nền sáng: `cap(2 × tỷ_lệ_sáng / 0.55 − 1)`, nhân với `cap(tỷ_lệ_chi_tiết_tối / 0.03)`. Nền trắng chiếm ít hơn 27.5% vùng tâm không được tính là nền trắng; chữ trắng rải rác trên mũ/quảng cáo vì vậy khó đạt hơn.
- Dải trắng: tìm hướng chính của pixel sáng bằng hiệp phương sai, đo dải theo hướng đó để chịu biển nghiêng. Lấy chênh lệch tỷ lệ sáng giữa dải giữa và hai dải bên / 0.70, nhân với tỷ lệ đỏ nhỏ hơn của hai dải bên / 0.65. Màu trắng sáng đều hoặc chữ rải rác không tự tạo thành dải trắng.
- Nền xanh: tỷ lệ xanh trong tâm / 0.45.
- Ký hiệu sáng trên xanh: lấy diện tích thành phần sáng liên thông lớn nhất trong tâm / diện tích vùng tâm; điểm ký hiệu là `cap(tỷ_lệ_này / 0.16)²`. Nhân với độ tin cậy màu xanh `trung_vị_S_xanh / 255` trước khi so với phương án có vành đỏ. Không cộng các đốm sáng rời trên lá cây thành một ký hiệu lớn. Bình phương giúp giảm điểm mạnh khi bằng chứng ký hiệu yếu; độ bão hòa thấp giảm độ tin cậy của vùng xanh xám trên cây. Ký hiệu mảnh, đứt nhiều mảnh hoặc xanh bạc màu có thể bị nhận muộn hơn.
- Vành đỏ ngoài nền xanh: tỷ lệ đỏ trong khoảng 0.90–1.40 bán kính / 0.16.

Mỗi tỷ số được giới hạn 0–1. Lấy cấu trúc phù hợp nhất cho mỗi màu. Nếu một cách giải thích không đạt support/pattern tối thiểu nhưng cách còn lại đạt, giữ cách đạt; không loại ứng viên chỉ vì cách giải thích điểm thô cao hơn bị thiếu bằng chứng.

Cần `support ≥ 0.58` và `pattern ≥ 0.28`. Đây là hai điều kiện bằng chứng tối thiểu, giúp tránh nhận một mảng màu đơn thuần hoặc ngôi sao vàng trên nền cờ đỏ như biển.

### 3.6. Chấm điểm chung

```text
score = 0.40 × shape
      + 0.25 × support
      + 0.25 × pattern
      + 0.10 × isolation
```

Ngưỡng chung tăng nhẹ, liên tục khi bán kính gần mức tối thiểu:

```text
small = 1 − cap((r / r_min − 1) / 2)
required_score = MIN_SIGN_SCORE + SMALL_SIGN_SCORE_BONUS × small
```

Với mặc định, ngưỡng từ 0.81 ở mức bán kính tối thiểu giảm dần về 0.78 khi r ≥ 3r_min. Không có ngưỡng riêng cho từng loại biển hoặc video. Điểm là mức phù hợp theo quy tắc, **không phải xác suất đúng**.

Các điều kiện bắt buộc gồm vị trí, kích thước, không gần chữ nhật, tỷ lệ trục, sai lệch elip và bằng chứng tối thiểu. Những tiêu chí còn lại đóng góp vào điểm; không có thêm chuỗi điều kiện riêng theo từng biển.

### 3.7. Bounding box và loại trùng

Box được tính từ kích thước elip sau xoay, thêm lề bằng 8% bán kính trục lớn và giới hạn trong frame.

`suppress_duplicates()` loại box nhỏ nếu ít nhất 85% diện tích của nó nằm trong một box lớn đã vượt validator (`NMS_CONTAINMENT`). Sau đó giữ ứng viên điểm cao khi IoU ≥ 0.35. Nếu có một box ngoài đồng tâm, diện tích không quá hai lần box đã giữ và điểm chỉ thấp hơn tối đa 0.10, ưu tiên box ngoài để bớt cắt mất vành đỏ. Bước chứa nhau xử lý trường hợp box nhỏ nằm trong biển nhưng IoU rất thấp; các biển cạnh nhau không bị gộp chỉ vì ở chung cột.

## 4. Tracker giảm nháy

| Nội dung | Chính sách mặc định |
| --- | --- |
| Xác nhận | Khoảng 4 lần detector nhận trong cửa sổ 6 frame ở 30 FPS; điểm trung bình ≥ `MIN_SIGN_SCORE`. Quy đổi số frame theo FPS. |
| Ghép | Khoảng cách tâm theo kích thước box và tỷ lệ kích thước; ghép một–một, ưu tiên khoảng cách chuẩn hóa nhỏ. |
| Dự đoán | Tâm, chiều rộng, chiều cao và vận tốc gần đây; vận tốc được giới hạn. |
| Làm mượt | Tâm dùng 35% dự đoán + 65% quan sát; kích thước dùng trực tiếp số đo mới để tránh box nhỏ hơn biển khi tiến lại gần. |
| Mất detector | Optical flow hợp lệ tiếp tục là một quan sát ảnh. Nếu cả detector và flow không còn xác minh được mục tiêu, giữ bằng dự đoán tối đa khoảng 0.25 giây. |
| Hết hạn | Xóa track sau khoảng 0.8 giây không có detector lẫn flow hợp lệ. Không xóa chỉ vì thiếu detector khi flow vẫn được xác minh. |

Optical flow Lucas–Kanade chỉ chạy cho track đã xác nhận và đang không ghép được detector. Tìm điểm trong box, theo dõi xuôi/ngược trên ROI cục bộ, cần ít nhất 4 điểm chuyển động nhất quán. Vị trí mới còn phải phù hợp màu/cấu trúc từ **cùng hàm appearance** của detector và có độ tương đồng ảnh tham chiếu ít nhất 0.70.

Sau dự đoán LK, dùng `matchTemplate` cục bộ với ảnh tham chiếu ở ba tỷ lệ 0.94/1.00/1.06 quanh kích thước dự đoán, tìm trong lề 15% kích thước box (tối thiểu 2 pixel). Chỉ giữ phương án vượt ngưỡng tương đồng 0.70 và bằng chứng màu/biên/ký hiệu. Bước này giảm sai lệch tâm/kích thước khi các điểm LK chỉ bám vào ký hiệu; đây là template của track lấy từ chính video, không phải thư viện mẫu biển hoặc mô hình huấn luyện sẵn.

`misses` đếm số frame thiếu detector để tính khoảng ghép đo; `unsupported` đếm số frame liên tiếp thiếu mọi bằng chứng ảnh hợp lệ để quyết định giữ/xóa box. Flow không tự tạo track mới và không cập nhật ảnh tham chiếu; ảnh tham chiếu được làm mới khi detector nhận lại mục tiêu, giúp hạn chế trôi tích lũy.

Ảnh tham chiếu được cắt từ box đo trực tiếp của detector, không từ box làm mượt có thể bị trễ. Flow không tạo biển mới và không được tính là lần detector nhận. Track chỉ tiếp tục khi quan sát ảnh vẫn vượt kiểm tra màu, biên và cấu trúc; khi thiếu bằng chứng ảnh, thời gian giữ/xóa vẫn hữu hạn như bảng trên. Khi detector nhận lại trước lúc hết hạn, track có thể hiện ngay. Box hiển thị được loại trùng lần nữa để giảm hai track cùng vẽ một biển.

Giữ box lâu hơn giúp che khoảng mất dấu nhưng cũng có thể kéo dài box sai. Vì vậy không dùng cách tăng thời gian giữ làm biện pháp duy nhất chống nháy.

## 5. Tham số nên chỉnh

| Nhóm | Tham số ở đầu file | Đánh đổi |
| --- | --- | --- |
| Vùng tìm | `CENTER_ROI`, `CONTEXT_MARGIN_RATIO` | Camera khác có thể cần vùng khác; ROI rộng có thêm nhiễu. |
| Khoảng cách/kích thước | `RADIUS_RANGE` | Giảm bán kính tối thiểu có thể nhận sớm và tăng box nhỏ sai. |
| Tròn/nghiêng | `MIN_AXIS_RATIO`, `MAX_ELLIPSE_ERROR`, `MIN_BOUNDARY_SUPPORT`, `MIN_LIGHT_NOTCH_RATIO`, `MIN_LIGHT_NOTCH_AREA_RATIO`, `RECTANGLE_FILL_LIMIT` | Giảm tỷ lệ điểm biên cần khớp hoặc tỷ lệ pixel sáng trong phần lõm có thể giữ biển bị ký hiệu cắt vào mask, nhưng thêm vật gần tròn. |
| Màu | `RED_HUE_RANGES`, `BLUE_HUE_RANGE`, `COLOR_MIN_SATURATION`, `COLOR_MIN_VALUE` | Mở rộng quá mức dễ nối biển vào nền. |
| Ánh sáng | `ENABLE_LIGHT_NORMALIZATION`, `CLAHE_CLIP_LIMIT`, `LIGHT_NORMALIZATION_BLEND` | Tăng mạnh có thể tăng nhiễu; không phục hồi được mọi màu sai. |
| Điểm | `SCORE_WEIGHTS`, `MIN_SIGN_SCORE`, `SMALL_SIGN_SCORE_BONUS` | Tăng ngưỡng giảm ứng viên yếu nhưng có thể nhận muộn. Tổng trọng số phải bằng 1. |
| Bằng chứng | `MIN_ANGULAR_SUPPORT`, `MIN_RADIAL_SUPPORT`, `MIN_PATTERN_SCORE`, `PATTERN_TARGETS`, `SYMBOL_SATURATION_RATIO` | Biên màu chặt hơn giảm mảng nền nhưng có thể bỏ vành mờ hoặc dính nền. Tỷ lệ bão hòa ký hiệu thấp hơn dễ bỏ ký hiệu bị lem màu. |
| Loại trùng | `NMS_IOU`, `NMS_CONTAINMENT` | Giảm ngưỡng chứa nhau có thể gộp nhầm hai biển che nhau; hiện dùng 85% diện tích box nhỏ. |
| Tracker | Các hằng số `TRACK_*`, `FLOW_MIN_SIMILARITY` | Xác nhận nhanh/giữ lâu có thể làm box sai xuất hiện hoặc tồn tại lâu hơn. |

Các vùng lấy mẫu, kích thước patch và định nghĩa đặc trưng là một phần của phương pháp. Không nên thay hàng loạt cùng lúc. Dùng một bộ cấu hình chung cho hai video và kiểm chứng lại sau mỗi thay đổi có mục đích.

Có thể xem bộ đếm loại ứng viên bằng một script ngoài:

```python
stats = {}
detector = SignDetector()
candidates = detector.detect(frame, diagnostics=stats)
print(stats)
```

`reject_geometry`, `reject_size_position`, `reject_appearance`, `reject_score` chỉ là số sự kiện loại trong pipeline; không phải số biển bị bỏ sót. `proposals` tính trước loại trùng, `selected` tính sau loại trùng.

## 6. Đọc code theo hàm

| Hàm/lớp | Nhiệm vụ |
| --- | --- |
| `color_masks()` | Hai định nghĩa màu dùng chung cho tạo ứng viên, đo đặc trưng và flow. |
| `split_color_components()` | Otsu cục bộ trên S để tách mặt biển bão hòa cao khỏi nền dính cùng màu. |
| `light_masks()` | Định nghĩa ký hiệu sáng dùng chung cho chấm điểm và kiểm tra vết lõm. |
| `boundary_features()`, `bar_evidence()`, `symbol_evidence()` | Xác minh biên và tách nền bằng cùng mặt nạ màu, đo dải trắng theo hướng chính và ký hiệu sáng liên thông. |
| `contour_geometry()` | Kiểm tra chữ nhật, khớp elip và đo sai lệch biên thật. |
| `ellipse_patch()` | Đưa elip nghiêng về vùng tròn chuẩn hóa. |
| `appearance_features()` | Đo vành màu, cấu trúc tâm và màu bên ngoài; chọn cách giải thích hợp lệ. |
| `SignDetector.detect()` | Cắt ROI, xử lý sáng/màu, tạo ứng viên và áp dụng một validator chung. |
| `circular_candidates()` | Hàm tiện lợi kiểm tra một frame; khi xử lý video sẽ tái sử dụng một detector. |
| `intersection_over_union()`, `suppress_duplicates()` | Tính giao/hợp và loại box trùng. |
| `SignTracker` | Ghép, xác nhận, làm mượt, flow khi mất dấu và hết hạn track. |
| `process_video()` | Đọc từng frame, nhận diện/theo dõi, vẽ và ghi video. |
| `main()` | Đọc cấu hình trong file, chạy hai video lần lượt. |

## 7. Đầu ra và kiểm tra

Mỗi nguồn tạo một MP4 trong `output`, dùng codec `mp4v`, giữ kích thước và FPS nguồn, ảnh màu và MSSV trên mỗi frame. VideoWriter không sao chép âm thanh.

| File | Số frame nguồn/đầu ra | Kích thước | FPS đọc lại |
| --- | --- | --- | --- |
| `video1_TEST.mp4` | 3087 / 3087 | 1920×1080 | 29.97 |
| `video2_TEST.mp4` | 2278 / 2278 | 1920×1080 | 29.97 |

Đã đọc toàn bộ hai đầu ra đến EOF, kiểm tra đủ frame và ảnh ba kênh. FPS nguồn khoảng 29.97002997; metadata MP4 đọc lại 29.97, sai khác dưới 0.001 FPS do biểu diễn thời gian của bộ ghi. Không ép nguồn thành 30 FPS.

`output/verification_manifest.json` ghi mốc xác minh cục bộ, mã băm code/nguồn/đầu ra và kết quả giải mã. Chỉ trạng thái `verified` mới ghi nhận bước kiểm tra đã xong. Đây là xác minh phiên bản và tính toàn vẹn file, không phải đánh giá độ chính xác nhận diện. Nếu sửa code hoặc ghi đè video cần đối chiếu lại mã băm. Manifest được tạo bởi bước kiểm tra ngoài chương trình; bấm Run không tự cập nhật manifest.

Bộ đọc/ghi được đóng trong `finally` khi thành công hoặc lỗi. Chế độ toàn bộ đọc đến EOF; nếu số frame không khớp metadata nguồn, chương trình báo lỗi và không in hoàn thành. File dở dang có thể còn sau khi lỗi và được ghi đè ở lần chạy thành công tiếp theo.

Các kiểm tra chức năng đã thực hiện cục bộ:

- Biển đỏ có dải trắng, biển xanh, elip nghiêng, ảnh tối và độ rộng 960/1920; detector không sửa frame nguồn.
- Loại bảng quảng cáo chữ nhật, cờ đỏ sao vàng và mảng đỏ/xanh nhạt không có ký hiệu trong các ảnh kiểm tra.
- Phân biệt một ký hiệu sáng liền với các đốm sáng rời có cùng tổng diện tích.
- Loại box nhỏ lồng trong box lớn ngay cả khi IoU thấp; giữ hai box của hai biển cạnh nhau.
- Theo đúng kích thước mới khi biển phóng to; flow đã xác minh giữ track qua khoảng detector mất dấu, rồi box ngừng hiện và track bị xóa khi mục tiêu biến mất.
- Đoạn biển xanh đầu video 1 trên đầu ra đầy đủ: trong 239 frame xác định được vị trí mục tiêu, bản cuối có box 232 frame, bắt đầu ở frame 7; khoảng mất box liên tục dài nhất sau khi xuất hiện là 0 frame. Bản trước lượt giảm nhầm có 234 frame và bắt đầu ở frame 5. Công cụ đo chỉ phục vụ hồi quy mục tiêu đã biết, không tham gia thuật toán và không phải bộ nhãn chuẩn.
- Kiểm tra riêng đoạn biển lớn gần giây 69 khi không còn nhận lại trong ROI; bước dò template cục bộ tăng nhẹ độ khớp với vùng vành đỏ đo từ ảnh trong đoạn thử. Không suy ra kết quả cho mọi biển.
- Đối chiếu đầu ra tại các vùng đã phản ánh ở giây 42/55/63/65/69/76/79/87/95 của video 1: trong các frame đối chiếu không còn box sai trên những vùng cửa hàng, mũ, cây và quảng cáo đó; vùng biển tại giây 69 còn một box, không còn box nhỏ lồng. Video 2 tại giây 65 vẫn bám biển xanh nhỏ trên nền xanh nhạt. Vùng đỏ rất xa tại giây 98 chưa đủ thông tin để kết luận là biển thật hay nhiễu; không tính việc không hiện box ở đó là bằng chứng sửa đúng.

Video, manifest, script kiểm tra, bản dự phòng và ảnh đối chiếu đều là dữ liệu cục bộ ngoài Git. Repository chỉ quản lý file Python chính và README.

## 8. Giới hạn

- Chưa có bộ nhãn thủ công hoặc đối chiếu đủ 24 biển; chưa kết luận điểm rubric, precision/recall hay bộ tham số tối ưu.
- Biển rất nhỏ, nhòe, bị che, thiếu màu hoặc thiếu đường biên vẫn có thể nhận muộn/mất dấu. Ký hiệu mảnh hoặc đứt nhiều mảnh, nền xanh bạc màu chịu ảnh hưởng của kiểm tra mới.
- Vật gần tròn có màu và họa tiết giống biển, mũ bảo hiểm hoặc logo tròn in trên quảng cáo vẫn có thể gây nhầm. Màu và hình dạng không đủ để phân biệt ngữ nghĩa trong mọi trường hợp.
- Chưa bảo đảm nhận đủ cả 5 biển từ xa ở mọi frame của cảnh đầu. Góc camera, ánh sáng và nền khác có thể cần đánh giá lại ROI và cấu hình chung.
- Tracker có thể giữ tạm một box sai nếu detector xác nhận sai. Flow/template xác minh bám ảnh, không chứng minh vật đó là biển thật. Box có thể còn lệch khi bị lóa, che hoặc biến dạng phối cảnh mạnh.
- Xử lý offline; tốc độ phụ thuộc CPU và độ phức tạp cảnh, chưa tối ưu cho thời gian thực.
- MSSV hiện là `TEST`; cần đổi thành MSSV thật trước khi xuất bản nộp. Báo cáo và Git push được để ngoài phạm vi lượt sửa code này.
