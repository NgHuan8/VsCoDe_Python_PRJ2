import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import config
from cores.dataset_loader import load_dlib_xml
from cores.model_builder import build_landmark_model

# ---------------------------------------------------------
# CUSTOM LOSS: TẬP TRUNG VÀO MẮT VÀ MIỆNG
# ---------------------------------------------------------
# Mảng 136 phần tử tương ứng với tọa độ X, Y của 68 điểm.
# Điểm 36-47 (Mắt) nằm ở index 72 đến 95.
# Điểm 48-67 (Miệng) nằm ở index 96 đến 135.
w_array = np.ones(136, dtype=np.float32)
w_array[72:136] = 5.0 # Tăng trọng số x5 cho vùng mắt và miệng
weights_tensor = tf.constant(w_array)

def custom_landmark_loss(y_true, y_pred):
    """ Hàm tính MSE nhưng nhân thêm trọng số để ép model học kỹ Mắt/Miệng """
    sq_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(sq_diff * weights_tensor)
# ---------------------------------------------------------

def main():
    print("Khởi tạo quá trình đọc dữ liệu...")
    X, y = load_dlib_xml(
        xml_path=config.XML_TRAIN_PATH, 
        image_base_dir=config.DATA_DIR, 
        target_size=(config.IMG_SIZE, config.IMG_SIZE)
    )
    print(f"\n=> TỔNG KẾT: Đã nạp thành công {X.shape[0]} ảnh lên RAM.")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    model = build_landmark_model(input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    
    # Thay 'mse' bằng hàm custom_landmark_loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=custom_landmark_loss, 
        metrics=['mae']
    )

    save_path = os.path.join(config.WEIGHTS_DIR, "face_landmark_68_best.keras")
    
    callbacks = [
        ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss', verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    print("BẮT ĐẦU HUẤN LUYỆN MODEL LANDMARK 68 ĐIỂM (TỐI ƯU MẮT/MIỆNG)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks
    )
    print(f"HOÀN TẤT HUẤN LUYỆN! Model tốt nhất đã được lưu tại: {save_path}")

if __name__ == "__main__":
    main()