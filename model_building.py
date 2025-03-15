import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_efficient_net(image_size=224, num_classes=4):
    """
    Xây dựng mô hình dựa trên EfficientNetV2-B0 với các kỹ thuật nâng cao
    """
    # Load EfficientNetV2-B0 pretrained
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size, image_size, 3)
    )
    
    # Đóng băng các layer của base model
    base_model.trainable = False
    
    # Xây dựng model
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dropout và BatchNorm để giảm overfitting
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    # Dense layers với regularization
    x = layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.01)
    )(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.01)
    )(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer với label smoothing
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="predictions"
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="efficient_net_mushroom")
    
    return model

def create_model(model_type="efficient_net", image_size=224, num_classes=4):
    """
    Tạo mô hình với các tùy chọn khác nhau
    """
    model_types = {
        "efficient_net": build_efficient_net,
    }
    
    if model_type not in model_types:
        raise ValueError(f"Không hỗ trợ loại mô hình: {model_type}")
    
    model = model_types[model_type](image_size=image_size, num_classes=num_classes)
    
    # Compile với các tùy chọn nâng cao
    optimizer = keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=0.0001,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc'),
            keras.metrics.CategoricalCrossentropy(name='cross_entropy')
        ]
    )
    
    return model

if __name__ == "__main__":
    model = create_model()
    model.summary()
    print("Xây dựng mô hình thành công!")