# Nhận diện biển báo giao thông hình tròn trong video

Bài giữa kỳ môn Xử lý ảnh số. Chương trình dùng OpenCV và NumPy để đọc từng khung hình, tìm ứng viên biển báo tròn, chấm điểm và theo dõi bounding box qua thời gian. Cùng một mã nguồn xử lý `video1.mp4` và `video2.mp4`; không dùng mô hình học máy huấn luyện sẵn.

## Chuẩn bị

- Python 3.11 hoặc phiên bản tương thích.
- Cài thư viện: `python -m pip install opencv-python numpy`.
- Đặt `video1.mp4` và `video2.mp4` cùng thư mục với `detect_circular_signs.py`.

## Chạy chương trình

Mở `detect_circular_signs.py`, chỉnh các hằng số ở đầu file:

```python
STUDENT_ID = "TEST"       # thay bằng MSSV hoặc các MSSV nối bằng dấu _
MAX_SECONDS = None        # None: toàn bộ video; 20: xem thử 20 giây
INPUT_VIDEOS = ("video1.mp4", "video2.mp4")
OUTPUT_FOLDER = "output"
```

Sau đó chạy:

```powershell
python detect_circular_signs.py
```

Khi `MAX_SECONDS = None`, kết quả nằm ở `output/video1_<MSSV>.mp4` và `output/video2_<MSSV>.mp4`. Video giữ FPS, độ phân giải và màu của file nguồn; MSSV được ghi trên từng khung hình. Chương trình báo lỗi nếu không mở được video hoặc giải mã dừng quá sớm.

## Nguyên lý

1. Làm mờ nhẹ, đổi BGR sang HSV và tạo mặt nạ màu đỏ/xanh.
2. Làm sạch mặt nạ, tìm contour và đôi khi dùng `HoughCircles` để khôi phục biển bị dính nền.
3. Lọc theo diện tích, tỷ lệ rộng/cao, độ tròn và bán kính. Bán kính tối thiểu hiện là `max(12, 0.0075 × chiều rộng ảnh)` pixel, khoảng 14,4 pixel ở độ rộng 1920. Ngưỡng này bỏ các box nhỏ nhưng cũng có thể làm mất biển thật ở xa hoặc các mảnh màu bên trong biển mà chưa khôi phục được cả biển.
4. Lọc vị trí: tâm ứng viên nằm trong khoảng 18–48% chiều cao ảnh và từ 12% chiều rộng ảnh trở sang phải. Đây là quy tắc theo tọa độ ảnh, chưa phải phân đoạn lòng đường.
5. Chấm điểm hình dạng, vành màu, tương phản trong/ngoài, vùng trung tâm và vị trí. Ứng viên phải đạt điểm gốc ít nhất `0.75`. Bản thử mới chia vành thành 12 phần: cần ít nhất 8 phần có màu với nhóm đỏ, 9 phần với nhóm xanh. Kiểm tra thêm tâm sáng của biển viền đỏ, dải sáng ngang của biển nền đỏ, hoặc ký hiệu sáng/viền đỏ của biển nền xanh. Điểm cuối gồm 80% điểm gốc và 20% điểm cấu trúc; vẫn phải đạt `0.75`. Các box trùng bị loại với ngưỡng IoU `0.35`.
6. Với biển viền đỏ, nền xanh, tìm riêng contour của vành đỏ và khớp ellipse bằng `fitEllipse`. Yêu cầu tỷ lệ trục ít nhất `0.70`, sai lệch biên ở phân vị 90% không quá `0.18`, màu đỏ hiện diện ở ít nhất 9/12 phần vành, nền xanh chiếm ít nhất 20% các mẫu bên trong và ít màu đỏ bên ngoài. Nhánh này dùng điểm riêng với ngưỡng `0.80`. Khi đã tìm được cả biển, bỏ những box nhỏ nằm gần hoàn toàn bên trong nó.
7. Theo dõi ứng viên giữa các khung hình. Ở khoảng 30 FPS, box được xác nhận khi phát hiện ít nhất 4 trong 6 khung hình gần nhất và điểm trung bình các lần phát hiện đạt `0.78`. Dự đoán tâm, chiều rộng và chiều cao theo vận tốc gần đây; cập nhật bằng 35% trạng thái dự đoán và 65% quan sát mới. Vận tốc được giới hạn để tránh nhảy quá xa; khi mất phát hiện, vận tốc giảm dần và box chỉ được giữ khoảng 0,3 giây.

Các ngưỡng chung nằm trong `detect_circular_signs.py`. Tăng bán kính tối thiểu giúp giảm box thừa từ chi tiết nhỏ nhưng có thể bỏ sót biển thật ở xa.

Để chẩn đoán, truyền một dictionary vào `circular_candidates(frame, diagnostics=stats)` hoặc `process_video(..., diagnostics=stats)`. Các bộ đếm cho biết ứng viên bị loại bởi kích thước, vị trí, bằng chứng màu, điểm số, số phần vành đạt yêu cầu, hoặc kiểm tra ellipse. Đây là số lần qua từng bước, không phải số biển thật; một ứng viên có thể được tạo từ nhiều mặt nạ.

## Mã nguồn trong repository

- `detect_circular_signs.py`: chương trình chính tạo hai video kết quả.
- `detect_circles_only.py`: thử nghiệm riêng bước tìm hình tròn.
- `explain_detection.py`: mã hỗ trợ giải thích các bước xử lý.

## Tình trạng và việc cần làm trước khi nộp

- Bản video hiện tại đã được xuất đủ 3.087 khung hình cho video 1 và 2.278 khung hình cho video 2, ở 1920×1080 và khoảng 29,97 FPS. Các file mang nhãn `TEST` để kiểm tra, chưa phải nhãn nộp bài.
- Chưa có bảng đối chiếu thủ công đủ 24 biển để tính độ chính xác theo rubric. Vẫn có thể có biển bị bỏ sót hoặc box nhầm.
- Bản thử kiểm tra cấu trúc vành được đánh giá riêng trên các khung hình mẫu và video xem thử. Hai video đầy đủ `video1_TEST.mp4`, `video2_TEST.mp4` hiện thuộc bản trước khi thêm bước này. Các bản thử mới có hậu tố `_structure_preview_20s`; đoạn so sánh có tên `video1_compare_structure_56-70s.mp4`.
- Sau đó bán kính tối thiểu được tăng từ 8,6 lên 14,4 px ở độ rộng 1920. Đoạn thử riêng cho thay đổi này: `video1_TEST_size_filter_60-70s.mp4`; các video thử cấu trúc trước đó vẫn dùng ngưỡng cũ.
- Bản cải tiến ellipse và dự đoán chuyển động được xuất riêng thành `video1_TEST_improved.mp4`, `video2_TEST_improved.mp4`. Kiểm tra 59 khung hình cách nhau 3 giây trên hai video bổ sung 5 lần phát hiện; kiểm tra trực quan cả 5 là biển viền đỏ, nền xanh. Đây không phải 5 biển khác nhau và chưa phải phép đo precision/recall toàn video. Log mẫu: `output/improvement_sample_audit.json`; log toàn video: `output/video1_improved_diagnostics.json`, `output/video2_improved_diagnostics.json`.
- Repository này chỉ chứa mã nguồn, README và `.gitignore`. Video nguồn, video kết quả, đề và báo cáo được giữ ngoài Git.
