import os

# Đường dẫn cơ sở
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'ibug_300W_large_face_landmark_dataset')

# Đường dẫn file XML
XML_TRAIN_PATH = os.path.join(DATA_DIR, 'labels_ibug_300W_train.xml')
XML_TEST_PATH = os.path.join(DATA_DIR, 'labels_ibug_300W_test.xml')

# Nơi lưu model
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True) # Tự động tạo thư mục nếu chưa có

# Siêu tham số
IMG_SIZE = 112
BATCH_SIZE = 32
EPOCHS = 100
PADDING_RATIO = 0.15  