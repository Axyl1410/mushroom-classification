import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from tqdm import tqdm

from data_preprocessing import read_labels, preprocess_image, create_datasets
from model_building import Patches, PatchEncoder, create_model

# Thiết lập thông số
IMG_SIZE = 128
MODEL_PATH = "models/mushroom_classifier_vit_best.h5"
TEST_DIR = "test"
LABELS_FILE = "mushroom_labels.csv"
SUBMISSION_FILE = "mushroom_predictions.csv"
CLASS_NAMES = ["Nấm mỡ", "Nấm bào ngư", "Nấm đùi gà", "Nấm linh chi trắng"]

# Thiết lập GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Đang sử dụng {len(physical_devices)} GPU...")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("Không tìm thấy GPU, đang sử dụng CPU...")

# Tải mô hình
def load_model(model_path):
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder}
        )
        print("Tải mô hình thành công!")
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {str(e)}")
        return create_model("hybrid_vit_cnn", IMG_SIZE, len(CLASS_NAMES))

# Đánh giá trên tập validation
def evaluate_on_validation(model, validation_dataset):
    print("Đánh giá trên tập validation...")
    evaluation = model.evaluate(validation_dataset, verbose=1)
    metrics = model.metrics_names
    for i, metric in enumerate(metrics):
        print(f"{metric}: {evaluation[i]:.4f}")
    return evaluation

# Thu thập dự đoán
def collect_predictions(model, dataset):
    all_labels, all_predictions = [], []
    for images, labels in tqdm(dataset, desc="Đang dự đoán"):
        batch_predictions = model.predict(images, verbose=0)
        all_labels.extend(labels.numpy())
        all_predictions.extend(batch_predictions)
    return np.array(all_labels), np.array(all_predictions)

# Ma trận nhầm lẫn
def predict_and_confusion_matrix(model, validation_dataset):
    print("Tạo ma trận nhầm lẫn...")
    validation_labels, predictions = collect_predictions(model, validation_dataset)
    predicted_classes = np.argmax(predictions, axis=1)
    cm = confusion_matrix(validation_labels, predicted_classes, labels=range(len(CLASS_NAMES)))
    class_report = classification_report(
        validation_labels, predicted_classes, target_names=CLASS_NAMES, output_dict=True
    )
    print("Báo cáo phân loại:")
    print(classification_report(validation_labels, predicted_classes, target_names=CLASS_NAMES))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()
    return cm, class_report

# Dự đoán trên tập test
def predict_on_test(model, test_dir, submission_file):
    print(f"Dự đoán trên tập test từ {test_dir}...")
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    test_files.sort()
    results = []
    
    for file_name in tqdm(test_files, desc="Dự đoán"):
        image_path = os.path.join(test_dir, file_name)
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = preprocess_image(img)
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        file_id = os.path.splitext(file_name)[0]
        results.append({'id': file_id, 'type': int(predicted_class)})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(submission_file, index=False)
    print(f"Đã lưu kết quả vào: {submission_file}")
    return results_df

# Quy trình đánh giá
def run_evaluation_pipeline():
    try:
        labels_df = read_labels(LABELS_FILE)
        _, validation_dataset, _, test_paths = create_datasets(
            labels_df, "train", "test", img_size=IMG_SIZE, batch_size=32
        )
        model = load_model(MODEL_PATH)
        evaluate_on_validation(model, validation_dataset)
        predict_and_confusion_matrix(model, validation_dataset)
        predict_on_test(model, TEST_DIR, SUBMISSION_FILE)
        print("Hoàn thành đánh giá!")
    except Exception as e:
        print(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    run_evaluation_pipeline()