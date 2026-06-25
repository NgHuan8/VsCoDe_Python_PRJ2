# core/model.py
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization

def build_landmark_model(input_shape=(112, 112, 3)):
    backbone = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    for layer in backbone.layers[:100]:
        layer.trainable = False

    model = Sequential([
        backbone,
        Flatten(),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(136, activation='linear') 
    ])
    
    return model