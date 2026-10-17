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
3. Lọc theo diện tích, tỷ lệ rộng/cao, độ tròn và bán kính. Bán kính tối thiểu hiện là `max(8, 0.0045 × chiều rộng ảnh)` pixel, khoảng 8,6 pixel ở độ rộng 1920.
4. Lọc vị trí: tâm ứng viên nằm trong khoảng 18–48% chiều cao ảnh và từ 12% chiều rộng ảnh trở sang phải. Đây là quy tắc theo tọa độ ảnh, chưa phải phân đoạn lòng đường.
5. Chấm điểm hình dạng, vành màu, tương phản trong/ngoài, vùng trung tâm và vị trí. Giữ ứng viên đạt ít nhất `0.75`, rồi bỏ các box trùng với ngưỡng IoU `0.35`.
6. Theo dõi ứng viên giữa các khung hình. Box xuất hiện sau khoảng 3 lần nhận diện liên tiếp, được làm mượt theo tỷ lệ 70% vị trí cũ và 30% vị trí mới, và được giữ tối đa khoảng 0,3 giây khi tạm thời mất phát hiện.

Các ngưỡng chung nằm trong `detect_circular_signs.py`. Tăng bán kính tối thiểu giúp giảm box thừa từ chi tiết nhỏ nhưng có thể bỏ sót biển thật ở xa.

## Mã nguồn trong repository

- `detect_circular_signs.py`: chương trình chính tạo hai video kết quả.
- `detect_circles_only.py`: thử nghiệm riêng bước tìm hình tròn.
- `explain_detection.py`: mã hỗ trợ giải thích các bước xử lý.

## Tình trạng và việc cần làm trước khi nộp

- Bản video hiện tại đã được xuất đủ 3.087 khung hình cho video 1 và 2.278 khung hình cho video 2, ở 1920×1080 và khoảng 29,97 FPS. Các file mang nhãn `TEST` để kiểm tra, chưa phải nhãn nộp bài.
- Chưa có bảng đối chiếu thủ công đủ 24 biển để tính độ chính xác theo rubric. Vẫn có thể có biển bị bỏ sót hoặc box nhầm.
- Repository này chỉ chứa mã nguồn, README và `.gitignore`. Video nguồn, video kết quả, đề và báo cáo được giữ ngoài Git.
