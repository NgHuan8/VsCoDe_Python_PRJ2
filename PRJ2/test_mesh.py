import cv2
import mediapipe as mp
print(mp.__file__)

# 1. Khởi tạo các công cụ vẽ và giải pháp Face Mesh của MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Cấu hình các tham số cho Face Mesh
# - max_num_faces=1: Chỉ tập trung xử lý 1 khuôn mặt (tài xế) để tăng tốc độ
# - refine_landmarks=True: Bắt buộc bật để trích xuất thêm tọa độ chi tiết của đồng tử mắt
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Mở kết nối với Webcam (0 là camera mặc định của máy tính)
cap = cv2.VideoCapture(0)

print("Hệ thống đang khởi động webcam... Nhấn 'q' trên cửa sổ video để THOÁT.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Không thể đọc dữ liệu từ webcam. Vui lòng kiểm tra lại thiết bị.")
        break

    # Lật ảnh theo chiều ngang để tạo hiệu ứng soi gương (giúp tài xế nhìn tự nhiên hơn)
    frame = cv2.flip(frame, 1)

    # OpenCV đọc ảnh ở dạng BGR, nhưng MediaPipe yêu cầu định dạng RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Đưa khung hình vào mô hình AI của MediaPipe để tính toán tọa độ
    results = face_mesh.process(rgb_frame)

    # 3. Vẽ lưới điểm đặc trưng lên khuôn mặt nếu tìm thấy
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Vẽ các đường lưới kết nối tổng thể trên toàn khuôn mặt (màu trắng mảnh)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Vẽ nổi bật đường viền của Mắt, Lông mày và Môi (màu xanh/đỏ)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Vẽ vòng tròn bao quanh đồng tử mắt (Chỉ có khi bật refine_landmarks=True)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

    # 4. Hiển thị khung hình đã vẽ lưới điểm lên màn hình
    cv2.imshow('MediaPipe Face Mesh - Test Pipeline', frame)

    # Dừng chương trình ngay lập tức khi người dùng ấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Giải phóng tài nguyên phần cứng sau khi tắt chương trình
cap.release()
cv2.destroyAllWindows()
face_mesh.close()