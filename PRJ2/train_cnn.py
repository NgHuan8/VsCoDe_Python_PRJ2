import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (Chỉ đường dẫn tới các folder giải nén)
# ==========================================
TRAIN_DIR = 'C:\\Users\\Asus Zenbook14X OLED\\Documents\\VsCoDe_Python_PRJ2\\PRJ2\\data\\train'  
VAL_DIR = 'C:\\Users\\Asus Zenbook14X OLED\\Documents\\VsCoDe_Python_PRJ2\\PRJ2\\data\\val'
IMG_SIZE = 48
BATCH_SIZE = 64

# ==========================================
# 2. TẠO LUỒNG TẢI DỮ LIỆU TỰ ĐỘNG TỪ THƯ MỤC (Data Generators)
# ==========================================
# Data Augmentation cho tập Train để chống Overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Chuẩn hóa pixel về [0, 1]
    rotation_range=15,          # Xoay ảnh ngẫu nhiên 15 độ
    zoom_range=0.1,             # Phóng to/thu nhỏ ngẫu nhiên 10%
    width_shift_range=0.1,      # Dịch chuyển ngang ngẫu nhiên
    height_shift_range=0.1      # Dịch chuyển dọc ngẫu nhiên
)

# Tập Validation CHỈ chuẩn hóa pixel, KHÔNG biến đổi ảnh (augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

print("Đang nạp dữ liệu Train...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',     # Chuyển thẳng về ảnh xám 1 kênh màu
    batch_size=BATCH_SIZE,
    class_mode='binary'         # Phân loại 2 nhãn (Đóng/Mở)
)

print("Đang nạp dữ liệu Validation...")
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ==========================================
# 3. XÂY DỰNG MẠNG CNN
# ==========================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(1, activation='sigmoid') # Lớp đầu ra 1 nơ-ron
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 4. HUẤN LUYỆN VÀ LƯU MÔ HÌNH
# ==========================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
]

print("\nBẮT ĐẦU HUẤN LUYỆN...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks
)

# Lưu mô hình
model.save("eye_state_classifier.keras")
print("Đã lưu mô hình thành công: eye_state_classifier.keras")

# In ra cách Keras tự động map nhãn để dùng cho code Realtime
print("\nTừ điển nhãn (Class Indices):", train_generator.class_indices)