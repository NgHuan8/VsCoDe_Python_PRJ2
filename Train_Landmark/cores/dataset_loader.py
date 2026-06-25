# core/dataset.py
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

def load_dlib_xml(xml_path, image_base_dir, target_size=(112, 112), padding_ratio=0.15):
    """
    Hàm đọc file XML chuẩn dlib (Kaggle), cắt ảnh và chuẩn hóa tọa độ siêu tốc.
    """
    print(f"Đang phân tích cấu trúc file XML: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    X = []
    y = []

    images = root.find('images').findall('image')
    print(f"Tìm thấy {len(images)} ảnh. Bắt đầu tiền xử lý dữ liệu...")

    for image_element in tqdm(images):
        file_name = image_element.get('file')
        img_path = os.path.join(image_base_dir, file_name)

        if not os.path.exists(img_path):
            continue

        box_element = image_element.find('box')
        if box_element is None:
            continue

        parts = box_element.findall('part')
        if len(parts) != 68:
            continue 

        landmarks = np.zeros((68, 2))
        valid_data = True
        
        for part in parts:
            idx = int(part.get('name'))
            if 0 <= idx <= 67:
                landmarks[idx][0] = float(part.get('x'))
                landmarks[idx][1] = float(part.get('y'))
            else:
                valid_data = False
                break
                
        if not valid_data:
            continue

        img = cv2.imread(img_path)
        if img is None: 
            continue
        h_img, w_img = img.shape[:2]

        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)

        box_w = x_max - x_min
        box_h = y_max - y_min

        pad_w = int(box_w * padding_ratio)
        pad_h = int(box_h * padding_ratio)

        x1 = max(0, int(x_min - pad_w))
        y1 = max(0, int(y_min - pad_h))
        x2 = min(w_img, int(x_max + pad_w))
        y2 = min(h_img, int(y_max + pad_h))

        final_box_w = x2 - x1
        final_box_h = y2 - y1

        if final_box_w <= 0 or final_box_h <= 0: 
            continue

        face_crop = img[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, target_size)
        
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = (face_rgb.astype(np.float32) / 127.5) - 1.0

        norm_landmarks = np.zeros_like(landmarks)
        for i in range(68):
            norm_landmarks[i][0] = (landmarks[i][0] - x1) / final_box_w
            norm_landmarks[i][1] = (landmarks[i][1] - y1) / final_box_h

        X.append(face_normalized)
        y.append(norm_landmarks.flatten())

    return np.array(X), np.array(y)