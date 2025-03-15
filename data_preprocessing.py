import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2

# Thiết lập thông số
IMG_SIZE = 224  # Kích thước ảnh phù hợp với EfficientNetV2
BATCH_SIZE = 16  # Giảm batch size để tăng độ ổn định
SEED = 42
TRAIN_DIR = "train"
TEST_DIR = "test"
LABELS_FILE = "mushroom_labels.csv"
VALIDATION_SPLIT = 0.2

# Thiết lập để sử dụng GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Tìm thấy {len(physical_devices)} GPU, đang thiết lập...")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Đã thiết lập GPU thành công!")
else:
    print("Không tìm thấy GPU, đang sử dụng CPU...")

# Đọc file CSV chứa nhãn
def read_labels(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Đọc được {len(df)} nhãn từ file {csv_path}")
    print(df.head())
    print("Phân phối nhãn:")
    print(df['type'].value_counts())
    return df

# Hàm tiền xử lý ảnh
def preprocess_image(image):
    """
    Tiền xử lý ảnh với các kỹ thuật nâng cao
    """
    image = tf.cast(image, tf.float32)
    
    # Chuẩn hóa ảnh theo cách EfficientNet yêu cầu
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    
    return image

def create_augmentation_layer():
    """
    Tạo một sequential layer cho data augmentation
    """
    return tf.keras.Sequential([
        # Xoay ảnh ngẫu nhiên bằng cách sử dụng affine transform
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomZoom(0.2),
        
        # Color augmentation
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
        
        # Gaussian noise
        tf.keras.layers.GaussianNoise(0.01)
    ])

def create_cutmix_labels(labels, alpha=0.2):
    """
    Tạo nhãn cho CutMix augmentation
    """
    batch_size = tf.shape(labels)[0]
    
    # Tạo ma trận one-hot từ labels
    labels_onehot = tf.one_hot(labels, depth=4)
    
    # Tạo random indices để trộn
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_labels = tf.gather(labels_onehot, indices)
    
    # Tạo random weights
    random_uniform = tf.random.uniform([batch_size], 0, 1)
    weights = tf.clip_by_value(random_uniform * (1 + alpha), 0, 1)
    weights = tf.reshape(weights, [batch_size, 1])
    
    # Mix labels
    mixed_labels = weights * labels_onehot + (1 - weights) * shuffled_labels
    
    return mixed_labels

def create_datasets(labels_df, train_dir, test_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=VALIDATION_SPLIT):
    if not os.path.exists(train_dir):
        raise ValueError(f"Thư mục train không tồn tại: {train_dir}")
    
    print(f"Tạo dataset từ thư mục: {train_dir}")
    
    # Khởi tạo test_dataset và test_paths với giá trị mặc định
    test_dataset = None
    test_paths = []
    
    try:
        # Tạo augmentation layer
        augmentation = create_augmentation_layer()
        
        # Tải dữ liệu huấn luyện và xác thực
        train_dataset_full = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=True,
            seed=SEED,
            validation_split=val_split,
            subset="training",
            label_mode='int'
        )
        
        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=False,
            seed=SEED,
            validation_split=val_split,
            subset="validation",
            label_mode='int'
        )
        
        # Lấy tên lớp (theo thứ tự bảng chữ cái: bao ngu xam trang:0, dui ga baby:1, linh chi trang:2, nam mo:3)
        class_names = train_dataset_full.class_names
        print(f"Tên các lớp: {class_names}")
        
        # Ánh xạ nhãn mặc định sang nhãn mong muốn
        # Mặc định: bao ngu xam trang:0, dui ga baby:1, linh chi trang:2, nam mo:3
        # Mong muốn: nam mo:0, bao ngu xam trang:1, dui ga baby:2, linh chi trang:3
        mapping_dict = {
            class_names.index('bao ngu xam trang'): 1,
            class_names.index('dui ga baby'): 2,
            class_names.index('linh chi trang'): 3,
            class_names.index('nam mo'): 0
        }
        
        def remap_labels(image, label):
            # Sử dụng tf.gather để ánh xạ nhãn
            new_label = tf.gather(list(mapping_dict.values()), label)
            return image, new_label
        
        def augment_and_mix(image, label):
            # Áp dụng augmentation
            augmented_image = augmentation(image, training=True)
            preprocessed_image = preprocess_image(augmented_image)
            
            # Tạo mixed labels
            mixed_labels = create_cutmix_labels(label)
            
            return preprocessed_image, mixed_labels
        
        # Áp dụng pipeline xử lý cho training
        train_dataset = train_dataset_full.map(
            remap_labels,
            num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            augment_and_mix,
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        # Áp dụng pipeline xử lý cho validation
        validation_dataset = validation_dataset.map(
            remap_labels,
            num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            lambda x, y: (preprocess_image(x), tf.one_hot(y, depth=4)),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        print(f"Số batch huấn luyện: {tf.data.experimental.cardinality(train_dataset)}")
        print(f"Số batch xác thực: {tf.data.experimental.cardinality(validation_dataset)}")
        
        # Xử lý test dataset nếu có
        if os.path.exists(test_dir):
            print(f"Xử lý thư mục test: {test_dir}")
            try:
                test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if test_files:
                    test_ds = tf.data.Dataset.from_tensor_slices(test_files)
                    
                    def read_image(file_path):
                        img = tf.io.read_file(file_path)
                        img = tf.image.decode_image(img, channels=3, expand_animations=False)
                        img = tf.image.resize(img, [img_size, img_size])
                        return img
                    
                    test_dataset = test_ds.map(
                        lambda x: (preprocess_image(read_image(x)), x),
                        num_parallel_calls=tf.data.AUTOTUNE
                    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                    
                    print(f"Tổng số ảnh test: {len(test_files)}")
                    test_paths = test_files
            except Exception as e:
                print(f"Lỗi khi xử lý test: {str(e)}")
        
        return train_dataset, validation_dataset, test_dataset, test_paths
        
    except Exception as e:
        print(f"Lỗi khi tạo dataset: {str(e)}")
        raise

if __name__ == "__main__":
    labels_df = read_labels(LABELS_FILE)
    train_dataset, validation_dataset, test_dataset, test_paths = create_datasets(labels_df, TRAIN_DIR, TEST_DIR)
    print("Hoàn thành tiền xử lý dữ liệu!")