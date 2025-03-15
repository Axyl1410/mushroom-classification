import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from tqdm import tqdm
import glob

# Định nghĩa đường dẫn
MODEL_PATH = r"D:\Testing\aio-hutech\mushroom_vit_model_final.h5"
TEST_DIR = r"D:\Testing\aio-hutech\test"
OUTPUT_FILE = 'mushroom_predictions.csv'  # File CSV đầu ra

# Định nghĩa kích thước ảnh
IMG_SIZE = 32  # Kích thước cố định như đã đề cập

# Định nghĩa lại các lớp tùy chỉnh giống như trong mã huấn luyện
# Định nghĩa lớp Patches - phải giống hệt như trong mã huấn luyện
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super(Patches, self).get_config()
        config.update({
            'patch_size': self.patch_size
        })
        return config

# Định nghĩa lớp PatchEncoder - phải giống hệt như trong mã huấn luyện
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim
        })
        return config

# Hàm tiền xử lý ảnh
def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize thành 32x32
        img = img / 255.0  # Chuẩn hóa về khoảng [0, 1]
        return img
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
        return None

# Hàm chính để dự đoán
def predict_mushroom_types():
    # Kiểm tra xem mô hình có tồn tại không
    if not os.path.exists(MODEL_PATH):
        print(f"Không tìm thấy mô hình tại {MODEL_PATH}. Vui lòng kiểm tra đường dẫn.")
        return
    
    # Kiểm tra xem thư mục test có tồn tại không
    if not os.path.exists(TEST_DIR):
        print(f"Không tìm thấy thư mục test tại {TEST_DIR}. Vui lòng kiểm tra đường dẫn.")
        return
    
    print("Đang tải mô hình...")
    
    # Sử dụng custom_object_scope để đăng ký các lớp tùy chỉnh
    custom_objects = {
        'Patches': Patches,
        'PatchEncoder': PatchEncoder
    }
    
    with keras.utils.custom_object_scope(custom_objects):
        model = keras.models.load_model(MODEL_PATH)
    
    print("Đã tải mô hình thành công!")
    
    # Lấy tất cả các ảnh trong thư mục test
    test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    if not test_images:
        print(f"Không tìm thấy ảnh JPG nào trong thư mục {TEST_DIR}")
        return
    
    print(f"Tìm thấy {len(test_images)} ảnh để phân loại.")
    
    # Chuẩn bị dữ liệu dự đoán
    test_ids = []
    X_test = []
    
    print("Đang tiền xử lý ảnh...")
    for img_path in tqdm(test_images):
        img_id = os.path.basename(img_path).split('.')[0]  # Lấy tên file không có phần mở rộng
        img = preprocess_image(img_path)
        
        if img is not None:
            test_ids.append(img_id)
            X_test.append(img)
    
    X_test = np.array(X_test)
    
    # Thực hiện dự đoán
    print("Đang thực hiện dự đoán...")
    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Tạo DataFrame kết quả
    results = pd.DataFrame({
        'id': test_ids,
        'type': pred_classes
    })
    
    # Lưu kết quả vào file CSV
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Đã lưu kết quả phân loại vào {OUTPUT_FILE}")
    
    # Hiển thị thông tin về số lượng dự đoán cho mỗi loại nấm
    class_counts = pd.Series(pred_classes).value_counts().sort_index()
    class_names = {
        0: "Nấm mỡ",
        1: "Nấm bào ngư",
        2: "Nấm đùi gà",
        3: "Nấm linh chi trắng"
    }
    
    print("\nKết quả phân loại:")
    for class_idx, count in class_counts.items():
        print(f"{class_names.get(class_idx, f'Loại {class_idx}')}: {count} ảnh")
    
    return results

# Thực thi hàm chính
if __name__ == "__main__":
    predict_mushroom_types()