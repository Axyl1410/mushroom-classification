import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import glob
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Định nghĩa các đường dẫn đến mô hình
OLD_MODEL_PATH = "D:/Project/Olympic AI HCMC/models 1/mushroom_vit_model_final.h5"
NEW_MODEL_PATH = "D:/Project/Olympic AI HCMC/mushroom_vit_model_final.h5"
TEST_DIR = "D:/Project/Olympic AI HCMC/test"
OUTPUT_FILE = 'model_comparison_results.csv'

# Định nghĩa các tham số
IMG_SIZE = 32  # Kích thước cố định
NUM_CLASSES = 4
PATCH_SIZE = 4  # Kích thước patch cho Vision Transformer

# Kiểm tra GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU is available!")
else:
    print("GPU is not available. Using CPU instead.")

# Định nghĩa các lớp tùy chỉnh để tải mô hình TensorFlow
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

# Hàm định dạng thời gian
def format_time(seconds):
    minutes = int(seconds // 60)
    seconds_remainder = seconds % 60
    if minutes > 0:
        return f"{minutes} phút {seconds_remainder:.3f} giây"
    else:
        return f"{seconds_remainder:.3f} giây"

# Hàm tải mô hình TensorFlow với các lớp tùy chỉnh
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình tại {model_path}")
    
    print(f"Đang tải mô hình từ {model_path}...")
    
    # Đăng ký các lớp tùy chỉnh
    custom_objects = {
        'Patches': Patches,
        'PatchEncoder': PatchEncoder
    }
    
    with keras.utils.custom_object_scope(custom_objects):
        model = keras.models.load_model(model_path)
    
    print(f"Đã tải mô hình thành công!")
    return model

# Hàm dự đoán với một mô hình
def predict_with_model(model, images, model_name="Model"):
    start_time = time.time()
    
    # Thực hiện dự đoán
    predictions = model.predict(images, verbose=1)
    pred_classes = np.argmax(predictions, axis=1)
    
    end_time = time.time()
    prediction_time = end_time - start_time
    
    print(f"{model_name} - Thời gian dự đoán: {format_time(prediction_time)}")
    print(f"{model_name} - Thời gian trung bình: {format_time(prediction_time / len(images))} / ảnh")
    print(f"{model_name} - Tốc độ: {len(images) / prediction_time:.2f} ảnh/giây")
    
    return pred_classes, predictions, prediction_time

# Hàm so sánh dự đoán của hai mô hình
def compare_predictions(old_preds, new_preds, test_ids):
    # Kiểm tra mức độ tương đồng giữa hai mô hình
    matching = old_preds == new_preds
    match_rate = np.mean(matching) * 100
    
    # Tạo DataFrame kết quả
    comparison_df = pd.DataFrame({
        'id': test_ids,
        'old_model_prediction': old_preds,
        'new_model_prediction': new_preds,
        'models_match': matching
    })
    
    # Tìm những ảnh có dự đoán khác nhau
    diff_predictions = comparison_df[~comparison_df['models_match']]
    
    print(f"\n=== Kết quả so sánh dự đoán ===")
    print(f"Tổng số ảnh: {len(comparison_df)}")
    print(f"Số ảnh dự đoán giống nhau: {sum(matching)}")
    print(f"Số ảnh dự đoán khác nhau: {len(diff_predictions)}")
    print(f"Tỷ lệ dự đoán trùng khớp: {match_rate:.2f}%")
    
    return comparison_df, diff_predictions

# Hàm tạo ma trận confusion giữa hai mô hình
def create_model_confusion_matrix(old_preds, new_preds, class_names):
    # Tạo ma trận confusion giữa hai mô hình
    cm = confusion_matrix(old_preds, new_preds)
    
    # Vẽ ma trận confusion
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('So sánh dự đoán giữa hai mô hình')
    plt.ylabel('Mô hình cũ (dự đoán)')
    plt.xlabel('Mô hình mới (dự đoán)')
    plt.tight_layout()
    plt.savefig('model_comparison_matrix.png')
    plt.close()
    
    print(f"Đã lưu ma trận so sánh vào file 'model_comparison_matrix.png'")
    return cm

# Hàm phân tích sự khác biệt chi tiết
def analyze_differences(diff_predictions, old_probs, new_probs, class_names):
    # Thêm thông tin về xác suất của mỗi lớp
    # Lấy 5 ảnh có sự khác biệt lớn nhất để phân tích chi tiết
    if len(diff_predictions) > 0:
        # Tính độ tin cậy của mỗi mô hình cho dự đoán của nó
        old_confidences = []
        new_confidences = []
        
        for idx, row in diff_predictions.iterrows():
            old_class = row['old_model_prediction']
            new_class = row['new_model_prediction']
            
            old_conf = old_probs[idx][old_class]
            new_conf = new_probs[idx][new_class]
            
            old_confidences.append(old_conf)
            new_confidences.append(new_conf)
        
        diff_predictions['old_model_confidence'] = old_confidences
        diff_predictions['new_model_confidence'] = new_confidences
        diff_predictions['confidence_diff'] = np.abs(np.array(new_confidences) - np.array(old_confidences))
        
        # Sắp xếp theo sự khác biệt về độ tin cậy
        diff_sorted = diff_predictions.sort_values('confidence_diff', ascending=False)
        
        print("\n=== Phân tích chi tiết các dự đoán khác nhau ===")
        print(f"5 ảnh có sự khác biệt lớn nhất về độ tin cậy:")
        
        for i, (idx, row) in enumerate(diff_sorted.head(5).iterrows()):
            old_class = int(row['old_model_prediction'])
            new_class = int(row['new_model_prediction'])
            
            print(f"{i+1}. Ảnh ID: {row['id']}")
            print(f"   Mô hình cũ: {class_names[old_class]} (độ tin cậy: {row['old_model_confidence']:.2%})")
            print(f"   Mô hình mới: {class_names[new_class]} (độ tin cậy: {row['new_model_confidence']:.2%})")
            print(f"   Chênh lệch: {row['confidence_diff']:.2%}")
        
        # Tạo biểu đồ phân phối độ tin cậy
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(old_confidences, bins=10, alpha=0.5, label='Mô hình cũ')
        plt.hist(new_confidences, bins=10, alpha=0.5, label='Mô hình mới')
        plt.title('Phân phối độ tin cậy khi có sự khác biệt')
        plt.xlabel('Độ tin cậy')
        plt.ylabel('Số lượng ảnh')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(old_confidences, new_confidences, alpha=0.5)
        plt.title('Độ tin cậy của hai mô hình')
        plt.xlabel('Độ tin cậy mô hình cũ')
        plt.ylabel('Độ tin cậy mô hình mới')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('confidence_distribution.png')
        plt.close()
        
        print(f"Đã lưu biểu đồ phân phối độ tin cậy vào file 'confidence_distribution.png'")
        
        return diff_sorted
    
    return diff_predictions

# Hàm vẽ biểu đồ so sánh hiệu suất
def plot_performance_comparison(old_time, new_time, num_images):
    # Tính thời gian trung bình mỗi ảnh
    old_avg_time = old_time / num_images
    new_avg_time = new_time / num_images
    
    # Tính tốc độ ảnh/giây
    old_speed = num_images / old_time
    new_speed = num_images / new_time
    
    # Tạo biểu đồ so sánh
    plt.figure(figsize=(15, 6))
    
    # Biểu đồ 1: Thời gian trung bình mỗi ảnh
    plt.subplot(1, 3, 1)
    bars = plt.bar(['Mô hình cũ', 'Mô hình mới'], [old_avg_time, new_avg_time], color=['#4C72B0', '#55A868'])
    plt.title('Thời gian trung bình mỗi ảnh (giây)')
    plt.ylabel('Thời gian (giây)')
    
    # Thêm giá trị lên mỗi cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                f'{height:.3f}s',
                ha='center', va='bottom')
    
    # Biểu đồ 2: Tốc độ xử lý
    plt.subplot(1, 3, 2)
    bars = plt.bar(['Mô hình cũ', 'Mô hình mới'], [old_speed, new_speed], color=['#4C72B0', '#55A868'])
    plt.title('Tốc độ xử lý (ảnh/giây)')
    plt.ylabel('Ảnh/giây')
    
    # Thêm giá trị lên mỗi cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Biểu đồ 3: Tổng thời gian dự đoán
    plt.subplot(1, 3, 3)
    bars = plt.bar(['Mô hình cũ', 'Mô hình mới'], [old_time, new_time], color=['#4C72B0', '#55A868'])
    plt.title(f'Tổng thời gian dự đoán {num_images} ảnh (giây)')
    plt.ylabel('Thời gian (giây)')
    
    # Thêm giá trị lên mỗi cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                f'{height:.2f}s',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.close()
    
    print(f"Đã lưu biểu đồ so sánh hiệu suất vào file 'performance_comparison.png'")

# Hàm phân tích phân phối dự đoán của từng mô hình
def analyze_prediction_distributions(old_preds, new_preds, class_names):
    # Đếm số lượng mỗi lớp
    old_counts = np.bincount(old_preds, minlength=NUM_CLASSES)
    new_counts = np.bincount(new_preds, minlength=NUM_CLASSES)
    
    # Tính tỷ lệ phần trăm
    total = len(old_preds)
    old_pcts = old_counts / total * 100
    new_pcts = new_counts / total * 100
    
    # In bảng phân phối
    print("\n=== Phân phối dự đoán của mỗi mô hình ===")
    print(f"{'Lớp':<20} {'Mô hình cũ':^15} {'Mô hình mới':^15} {'Chênh lệch':^15}")
    print("-" * 70)
    
    for i in range(NUM_CLASSES):
        diff = new_pcts[i] - old_pcts[i]
        diff_str = f"{diff:+.2f}%" if diff != 0 else "0.00%"
        print(f"{class_names[i]:<20} {old_counts[i]:>5} ({old_pcts[i]:.2f}%) {new_counts[i]:>5} ({new_pcts[i]:.2f}%) {diff_str:^15}")
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 6))
    
    # Biểu đồ cột so sánh số lượng
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, old_counts, width, label='Mô hình cũ')
    plt.bar(x + width/2, new_counts, width, label='Mô hình mới')
    
    plt.xlabel('Lớp')
    plt.ylabel('Số lượng ảnh')
    plt.title('So sánh phân phối dự đoán')
    plt.xticks(x, class_names)
    plt.legend()
    
    # Thêm nhãn số lượng lên mỗi cột
    for i, v in enumerate(old_counts):
        plt.text(i - width/2, v + 0.5, str(v), ha='center')
    
    for i, v in enumerate(new_counts):
        plt.text(i + width/2, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('prediction_distribution.png')
    plt.close()
    
    print(f"Đã lưu biểu đồ phân phối dự đoán vào file 'prediction_distribution.png'")

# Hàm chính để so sánh mô hình
def compare_models():
    # Kiểm tra tập test
    if not os.path.exists(TEST_DIR):
        print(f"Không tìm thấy thư mục test tại {TEST_DIR}")
        return
    
    # Bắt đầu đo thời gian
    total_start_time = time.time()
    
    # Tải mô hình cũ
    start_time = time.time()
    try:
        old_model = load_model(OLD_MODEL_PATH)
        old_model_load_time = time.time() - start_time
        print(f"Thời gian tải mô hình cũ: {format_time(old_model_load_time)}")
    except Exception as e:
        print(f"Lỗi khi tải mô hình cũ: {e}")
        return
    
    # Tải mô hình mới
    start_time = time.time()
    try:
        new_model = load_model(NEW_MODEL_PATH)
        new_model_load_time = time.time() - start_time
        print(f"Thời gian tải mô hình mới: {format_time(new_model_load_time)}")
    except Exception as e:
        print(f"Lỗi khi tải mô hình mới: {e}")
        return
    
    # Tải dữ liệu test
    test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    if not test_images:
        print(f"Không tìm thấy ảnh JPG nào trong thư mục {TEST_DIR}")
        return
    
    test_ids = []
    X_test = []
    
    print(f"\nTìm thấy {len(test_images)} ảnh để so sánh")
    print("Đang tiền xử lý ảnh...")
    
    for img_path in tqdm(test_images):
        img_id = os.path.basename(img_path).split('.')[0]
        img = preprocess_image(img_path)
        
        if img is not None:
            test_ids.append(img_id)
            X_test.append(img)
    
    X_test = np.array(X_test)
    print(f"Đã tiền xử lý {len(X_test)} ảnh")
    
    # Đặt tên các lớp
    class_names = {
        0: "Nấm mỡ",
        1: "Nấm bào ngư",
        2: "Nấm đùi gà",
        3: "Nấm linh chi trắng"
    }
    
    # Dự đoán với mô hình cũ
    print("\n=== Dự đoán với mô hình cũ ===")
    old_preds, old_probs, old_time = predict_with_model(old_model, X_test, "Mô hình cũ")
    
    # Dự đoán với mô hình mới
    print("\n=== Dự đoán với mô hình mới ===")
    new_preds, new_probs, new_time = predict_with_model(new_model, X_test, "Mô hình mới")
    
    # So sánh dự đoán
    comparison_df, diff_predictions = compare_predictions(old_preds, new_preds, test_ids)
    
    # Tạo ma trận confusion giữa hai mô hình
    cm = create_model_confusion_matrix(old_preds, new_preds, list(class_names.values()))
    
    # Phân tích sự khác biệt chi tiết
    diff_analysis = analyze_differences(diff_predictions, old_probs, new_probs, class_names)
    
    # Vẽ biểu đồ so sánh hiệu suất
    plot_performance_comparison(old_time, new_time, len(X_test))
    
    # Phân tích phân phối dự đoán
    analyze_prediction_distributions(old_preds, new_preds, list(class_names.values()))
    
    # Thống kê tổng hợp
    speedup = old_time / new_time if new_time > 0 else float('inf')
    
    print("\n=== Tóm tắt so sánh ===")
    print(f"Tổng số ảnh thử nghiệm: {len(X_test)}")
    print(f"Tỷ lệ dự đoán trùng khớp: {comparison_df['models_match'].mean()*100:.2f}%")
    
    if speedup > 1:
        print(f"Mô hình mới nhanh hơn {speedup:.2f}x so với mô hình cũ")
    else:
        print(f"Mô hình cũ nhanh hơn {1/speedup:.2f}x so với mô hình mới")
    
    # Lưu kết quả so sánh chi tiết vào CSV
    comparison_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nĐã lưu kết quả so sánh chi tiết vào file '{OUTPUT_FILE}'")
    
    # Kết thúc đo thời gian
    total_time = time.time() - total_start_time
    print(f"\nTổng thời gian chạy: {format_time(total_time)}")
    
    return comparison_df

# Hàm main
if __name__ == "__main__":
    # Hiển thị thông tin về mô hình
    print("=== So sánh hai mô hình TensorFlow ===")
    print(f"Mô hình cũ: {os.path.basename(OLD_MODEL_PATH)}")
    print(f"Mô hình mới: {os.path.basename(NEW_MODEL_PATH)}")
    print(f"Thư mục test: {TEST_DIR}")
    
    # Thực hiện so sánh
    compare_models()