import cv2
import numpy as np
from keras.models import load_model

# 1. NẠP MÔ HÌNH VÀ CÔNG CỤ
print("Đang nạp mô hình Landmark 68 điểm...")
model_path = "face_landmark_68.keras"
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Lỗi nạp mô hình: {e}")
    exit()

# Sử dụng Haar Cascade siêu nhẹ của OpenCV để tìm khuôn mặt (Bounding Box)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. KHỞI ĐỘNG WEBCAM
cap = cv2.VideoCapture(0)  # Thay số 0 thành 1, 2... nếu bạn dùng nhiều camera
print("Webcam đã sẵn sàng. Bấm phím 'q' để thoát.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    frame = cv2.flip(frame, 1) # Lật ảnh như gương
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- BƯỚC 1: TÌM KHUÔN MẶT ---
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    for (x, y, w, h) in faces:
# --- BẮT ĐẦU SỬA: ÉP KHUNG Haar Cascade ÔM SÁT ---
        # 1. Đẩy khung xuống dưới (bỏ bớt phần tóc/trán thừa)
        offset_y = int(h * 0.15) 
        # 2. Bóp nhẹ hai bên má vào trong
        offset_x = int(w * 0.05) 
        
        # Tạo tọa độ Box MỚI ôm sát hơn
        x_tight = x + offset_x
        y_tight = y + offset_y
        w_tight = int(w * 0.9)
        h_tight = int(h * 0.8) # Cắt bớt phần cằm/cổ thừa
        
        # Vẽ viền xanh dương theo khung Tight này để bạn dễ kiểm tra
        cv2.rectangle(frame, (x_tight, y_tight), (x_tight+w_tight, y_tight+h_tight), (255, 0, 0), 2)
        
        # Xóa bỏ tỷ lệ padding 15% cũ đi vì khung Haar bản chất đã to sẵn rồi
        x1, y1 = x_tight, y_tight
        x2, y2 = x_tight + w_tight, y_tight + h_tight
        
        final_box_w = x2 - x1
        final_box_h = y2 - y1
        
        if final_box_w <= 0 or final_box_h <= 0: continue
            
        # --- BƯỚC 2: CẮT KHUÔN MẶT VÀ CHUẨN HÓA ---
        face_crop = frame[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (112, 112))
        
        # Đổi hệ màu và chuẩn hóa y hệt lúc Train (MobileNetV2 dùng RGB dải [-1, 1])
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_norm = (face_rgb.astype(np.float32) / 127.5) - 1.0
        face_input = np.expand_dims(face_norm, axis=0)
        
        # --- BƯỚC 3: DỰ ĐOÁN TỌA ĐỘ BẰNG MẠNG CNN TỰ XÂY ---
        predictions = model.predict(face_input, verbose=0)[0]
        
        # --- BƯỚC 4: GIẢI MÃ TỌA ĐỘ VÀ VẼ LÊN MÀN HÌNH ---
        for i in range(68):
            x_norm = predictions[i * 2]
            y_norm = predictions[i * 2 + 1]
            
            # Nhân ngược tỷ lệ dải [0, 1] về kích thước thật trên khung hình
            real_x = int(x_norm * final_box_w + x1)
            real_y = int(y_norm * final_box_h + y1)
            
            # Vẽ các điểm mốc (Màu xanh lá)
            cv2.circle(frame, (real_x, real_y), 2, (0, 255, 0), -1)
            
            # (Tùy chọn) Đánh số thứ tự các điểm để kiểm tra
            # cv2.putText(frame, str(i), (real_x+2, real_y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)

    # Hiển thị
    cv2.imshow('Custom 68-Landmark Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()