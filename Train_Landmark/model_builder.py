# model_builder.py
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization

def build_landmark_model(input_shape=(112, 112, 3)):
    # Trích xuất đặc trưng hình ảnh bằng MobileNetV2
    backbone = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Đóng băng 100 lớp đầu tiên để chống phá vỡ các đặc trưng cơ bản đã học
    for layer in backbone.layers[:100]:
        layer.trainable = False

    # Xây dựng kiến trúc Regression
    model = Sequential([
        backbone,
        Flatten(),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Hàm kích hoạt linear giúp AI xuất ra con số không bị giới hạn
        Dense(136, activation='linear') 
    ])
    
    return model