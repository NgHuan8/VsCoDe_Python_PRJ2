import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from dataset_loader import load_dlib_xml
from model_builder import build_landmark_model

# ==========================================
# 1. Cấu hình Đường dẫn (Tự động dò tìm chuẩn tuyệt đối)
# ==========================================
# Lấy đường dẫn thư mục chứa chính file train.py này (thư mục Train_Landmark)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ghép nối đường dẫn an toàn tới thư mục chứa bộ dataset
DATA_DIR = os.path.join(CURRENT_DIR, "data", "ibug_300W_large_face_landmark_dataset")

# Đường dẫn chuẩn xác tới file XML
XML_TRAIN_PATH = os.path.join(DATA_DIR, "labels_ibug_300W_train.xml")
XML_TEST_PATH = os.path.join(DATA_DIR, "labels_ibug_300W_test.xml")

IMG_SIZE = 112
BATCH_SIZE = 32
EPOCHS = 100

def main():
    # 2. Nạp dữ liệu tự động từ file XML
    X, y = load_dlib_xml(XML_TRAIN_PATH, DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE))
    print(f"\n=> TỔNG KẾT: Đã nạp thành công {X.shape[0]} ảnh lên RAM.")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    # 3. Khởi tạo mô hình
    model = build_landmark_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Sử dụng Loss là MSE (Phạt nặng các điểm bị lệch xa) và metrics MAE (Đo khoảng cách lệch thực tế)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )

    # 4. Cấu hình cơ chế bảo vệ quá trình huấn luyện
    callbacks = [
        ModelCheckpoint("face_landmark_68.keras", save_best_only=True, monitor='val_loss', verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    # 5. Khởi chạy
    print("BẮT ĐẦU HUẤN LUYỆN MODEL LANDMARK 68 ĐIỂM...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    print("HOÀN TẤT HUẤN LUYỆN!")

if __name__ == "__main__":
    main()