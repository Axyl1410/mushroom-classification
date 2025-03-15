import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from data_preprocessing import read_labels, create_datasets
from model_building import create_model

# Thiết lập thông số
IMG_SIZE = 224  # Kích thước ảnh phù hợp với EfficientNetV2
BATCH_SIZE = 16  # Giảm batch size để tăng độ ổn định
EPOCHS = 100  # Tăng số epochs
LEARNING_RATE = 1e-4
MODEL_TYPE = "efficient_net"
NUM_CLASSES = 4
TRAIN_DIR = "train"
TEST_DIR = "test"
LABELS_FILE = "mushroom_labels.csv"
MODEL_SAVE_PATH = "models/mushroom_classifier_efficient"
EARLY_STOPPING_PATIENCE = 15  # Tăng patience
REDUCE_LR_PATIENCE = 8  # Tăng patience cho learning rate
FINE_TUNE_AT = 100  # Số layer cuối cùng để fine-tune

# Thiết lập GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Đang sử dụng {len(physical_devices)} GPU...")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("Không tìm thấy GPU, đang sử dụng CPU...")

def setup_model(model_type, num_classes, image_size):
    # Tắt cảnh báo TensorFlow
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Tạo model với các tùy chọn mặc định
    model = create_model(model_type, image_size, num_classes)
    
    return model

# Callbacks
def get_callbacks(model_save_path):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        # TensorBoard callback
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        ),
        
        # Model checkpoint cho weights
        keras.callbacks.ModelCheckpoint(
            model_save_path + "_best.weights.h5",
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1,
            save_weights_only=True
        ),
        
        # Model checkpoint cho toàn bộ model
        keras.callbacks.ModelCheckpoint(
            model_save_path + "_full_best.h5",
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        
        # Reduce learning rate
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1,
            min_delta=0.001
        ),
        
        # CSV Logger
        keras.callbacks.CSVLogger(
            model_save_path + "_training_log.csv",
            separator=',',
            append=True
        )
    ]
    return callbacks

def train_model(model, train_generator, validation_generator, callbacks):
    print("Bắt đầu huấn luyện giai đoạn 1 (frozen)...")
    history1 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS // 2,  # Train một nửa số epochs với frozen base model
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nBắt đầu fine-tuning...")
    # Mở khóa các layer cuối của base model
    base_model = model.layers[1]  # EfficientNetV2 là layer thứ 2
    base_model.trainable = True
    
    for layer in base_model.layers[:-FINE_TUNE_AT]:
        layer.trainable = False
    
    # Compile lại model với learning rate thấp hơn
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE/10,
            weight_decay=0.0001,
            clipnorm=1.0
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc'),
            keras.metrics.CategoricalCrossentropy(name='cross_entropy')
        ]
    )
    
    # Train tiếp với fine-tuning
    history2 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,  # Train full epochs với fine-tuning
        callbacks=callbacks,
        verbose=1
    )
    
    # Kết hợp histories
    combined_history = {}
    for key in history1.history.keys():
        combined_history[key] = history1.history[key] + history2.history[key]
    
    return combined_history, model

def plot_training_history(history, save_path=None):
    metrics = ['accuracy', 'loss', 'top2_acc', 'cross_entropy']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', size=16)
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        ax.plot(history[metric], label=f'Train {metric}')
        ax.plot(history[f'val_{metric}'], label=f'Val {metric}')
        ax.set_title(f'Model {metric.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "_training_history.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_model(model, model_save_path):
    # Lưu toàn bộ model
    model.save(model_save_path + "_final.h5")
    # Lưu weights
    model.save_weights(model_save_path + "_final.weights.h5")
    print(f"Đã lưu mô hình tại: {model_save_path}_final.h5")
    print(f"Đã lưu weights tại: {model_save_path}_final.weights.h5")

def evaluate_model(model, test_dataset, test_paths):
    if test_dataset is None:
        print("Không có dữ liệu test để đánh giá")
        return
    
    print("\nĐánh giá mô hình trên tập test:")
    predictions = model.predict(test_dataset)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # In kết quả chi tiết cho từng ảnh
    print("\nKết quả chi tiết:")
    for i, (pred, path) in enumerate(zip(predictions, test_paths)):
        confidence = np.max(pred) * 100
        predicted_class = predicted_classes[i]
        print(f"Ảnh {os.path.basename(path)}:")
        print(f"  - Dự đoán: Loại {predicted_class}")
        print(f"  - Độ tin cậy: {confidence:.2f}%")
        
        # In xác suất cho tất cả các lớp
        for class_idx, prob in enumerate(pred):
            print(f"  - Xác suất loại {class_idx}: {prob*100:.2f}%")
        print()

def run_training_pipeline():
    try:
        # Đọc dữ liệu và tạo datasets
        labels_df = read_labels(LABELS_FILE)
        train_generator, validation_generator, test_dataset, test_paths = create_datasets(
            labels_df, TRAIN_DIR, TEST_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE
        )
        
        # Tạo và huấn luyện model
        model = setup_model(MODEL_TYPE, NUM_CLASSES, IMG_SIZE)
        model.summary()
        
        callbacks = get_callbacks(MODEL_SAVE_PATH)
        history, trained_model = train_model(model, train_generator, validation_generator, callbacks)
        
        # Vẽ biểu đồ và lưu kết quả
        plot_training_history(history, MODEL_SAVE_PATH)
        save_model(trained_model, MODEL_SAVE_PATH)
        
        # Đánh giá model
        evaluate_model(trained_model, test_dataset, test_paths)
        
        print("Huấn luyện hoàn tất!")
        return trained_model
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return None

if __name__ == "__main__":
    run_training_pipeline()