# config.py
# Đây là module cấu hình chính cho dự án, chứa các hằng số và đường dẫn quan trọng
# để thu thập dữ liệu và huấn luyện mô hình CNN nhận diện trạng thái mắt.
import os

# Đường dẫn thư mục lưu dữ liệu ảnh mắt để train CNN
DATASET_PATH = "eye_dataset"
CLOSED_DIR = os.path.join(DATASET_PATH, "closed")
OPEN_DIR = os.path.join(DATASET_PATH, "open")

# Chỉ số các điểm mốc (Landmarks) trên Face Mesh chuẩn của Google
# Mỗi mắt gồm 6 điểm để tính toán chỉ số hình học EAR
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Vùng miệng gồm 8 điểm để tính chỉ số mở rộng MAR
MOUTH_INDICES = [78, 81, 13, 311, 308, 402, 14, 178]

# Cấu hình camera và ngưỡng thử nghiệm ban đầu
CAMERA_INDEX = 0
EAR_THRESHOLD = 0.22  # Dưới ngưỡng này mắt được coi là đóng
MAR_THRESHOLD = 0.60  # Vượt ngưỡng này nghĩa là đang ngáp