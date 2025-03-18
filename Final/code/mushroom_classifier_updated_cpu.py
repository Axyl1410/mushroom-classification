import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
from tensorflow.keras.utils import to_categorical
import random
from tqdm import tqdm
import glob
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import gaussian_filter, map_coordinates
import time  # Thêm thư viện để đo thời gian

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU is available!")
else:
    print("GPU is not available. Using CPU instead.")

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Define paths
BASE_DIR = '.'  # Current directory
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
CSV_FILE = os.path.join(BASE_DIR, 'mushroom_labels.csv')

# Define constants
IMG_SIZE = 32  # Fixed image size as mentioned
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
PATCH_SIZE = 4  # Size of patches for Vision Transformer
NUM_HEADS = 8  # Number of attention heads
TRANSFORMER_LAYERS = 6  # Number of transformer layers
PROJECTION_DIM = PATCH_SIZE * PATCH_SIZE * 3  # Dimension of patch embedding
MLP_UNITS = [
    PROJECTION_DIM * 2,  # First layer is larger (expansion)
    PROJECTION_DIM,      # Second layer matches the input dimension (projection)
]  # Size of the MLP layers
DROPOUT_RATE = 0.1

# Hàm định dạng thời gian thành chuỗi dễ đọc
def format_time(seconds):
    """Chuyển đổi thời gian từ giây thành định dạng dễ đọc (phút:giây.mili giây)"""
    minutes = int(seconds // 60)
    seconds_remainder = seconds % 60
    if minutes > 0:
        return f"{minutes} phút {seconds_remainder:.3f} giây"
    else:
        return f"{seconds_remainder:.3f} giây"

# Load labels from CSV
def load_labels():
    labels_df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(labels_df)} labeled images from CSV file")
    return labels_df

# Function to preprocess a single image
def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 32x32
        img = img / 255.0  # Normalize to [0, 1]
        return img
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

# Function to perform elastic deformation for data augmentation
def elastic_transform(image, alpha=36, sigma=4):
    """Apply elastic transformation on image"""
    random_state = np.random.RandomState(None)
    
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Make sure image is in correct format for transformation
    if image.shape[-1] != 3 and len(image.shape) == 3:
        # Convert grayscale to RGB if needed
        image = np.stack([image] * 3, axis=2)
    
    shape = image.shape
    
    # Generate displacement fields
    dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma) * alpha
    
    # Generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    
    # Apply displacement
    distorted_image = np.zeros_like(image)
    for i in range(image.shape[2]):
        distorted_image[..., i] = map_coordinates(image[..., i], indices, order=1).reshape(shape[:2])
    
    # Ensure values are in correct range
    distorted_image = np.clip(distorted_image, 0, 255).astype(np.uint8)
    
    return distorted_image

# Function to perform Mixup augmentation
def mixup_images(img1, img2, alpha=0.2):
    """Combine two images using mixup technique"""
    # Ensure both images are numpy arrays
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    
    # Ensure both images have the same shape
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Normalize images if they're not already (0-1 scale)
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    # Perform mixup
    lam = np.random.beta(alpha, alpha)
    mixed_img = lam * img1 + (1 - lam) * img2
    
    return mixed_img

# Function to perform CutMix augmentation
def cutmix_images(img1, img2, cut_size=None):
    """Combine two images using cutmix technique"""
    # Ensure both images are numpy arrays
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    
    # Ensure both images have the same shape
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Normalize images if they're not already (0-1 scale)
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    # Define size of the cut region
    h, w = img1.shape[0], img1.shape[1]
    if cut_size is None:
        cut_size = min(h, w) // 2
    
    # Random location for the cut
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    # Ensure the cut region stays within the image boundaries
    x1 = max(cx - cut_size // 2, 0)
    y1 = max(cy - cut_size // 2, 0)
    x2 = min(cx + cut_size // 2, w)
    y2 = min(cy + cut_size // 2, h)
    
    # Create the mixed image
    mixed_img = img1.copy()
    mixed_img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    
    return mixed_img

# Enhanced function to create data augmentation techniques
def augment_image_enhanced(img, file_prefix, save_dir=None, all_images=None):
    """Enhanced augmentation function with advanced techniques"""
    augmented_images = []
    augmented_paths = []
    
    # Convert numpy array to PIL Image
    if isinstance(img, np.ndarray):
        if img.max() <= 1.0:  # If normalized to [0,1], convert to [0,255]
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
        else:
            img_pil = Image.fromarray(img.astype(np.uint8))
    else:
        img_pil = img
    
    # 1. Rotation (90, 180, 270 degrees)
    for angle in [90, 180, 270]:
        rotated = img_pil.rotate(angle)
        if save_dir:
            save_path = os.path.join(save_dir, f"{file_prefix}_rot{angle}.jpg")
            rotated.save(save_path)
            augmented_paths.append(save_path)
        augmented_images.append(np.array(rotated) / 255.0)
    
    # 2. Horizontal and Vertical Flips
    flipped_h = ImageOps.mirror(img_pil)
    flipped_v = ImageOps.flip(img_pil)
    if save_dir:
        flipped_h.save(os.path.join(save_dir, f"{file_prefix}_fliph.jpg"))
        flipped_v.save(os.path.join(save_dir, f"{file_prefix}_flipv.jpg"))
        augmented_paths.extend([
            os.path.join(save_dir, f"{file_prefix}_fliph.jpg"),
            os.path.join(save_dir, f"{file_prefix}_flipv.jpg")
        ])
    augmented_images.extend([np.array(flipped_h) / 255.0, np.array(flipped_v) / 255.0])
    
    # 3. Brightness adjustment (increase and decrease)
    brightness_enhancer = ImageEnhance.Brightness(img_pil)
    brightened = brightness_enhancer.enhance(1.5)
    darkened = brightness_enhancer.enhance(0.7)
    if save_dir:
        brightened.save(os.path.join(save_dir, f"{file_prefix}_bright.jpg"))
        darkened.save(os.path.join(save_dir, f"{file_prefix}_dark.jpg"))
        augmented_paths.extend([
            os.path.join(save_dir, f"{file_prefix}_bright.jpg"),
            os.path.join(save_dir, f"{file_prefix}_dark.jpg")
        ])
    augmented_images.extend([np.array(brightened) / 255.0, np.array(darkened) / 255.0])
    
    # 4. Contrast adjustment
    contrast_enhancer = ImageEnhance.Contrast(img_pil)
    increased_contrast = contrast_enhancer.enhance(1.5)
    decreased_contrast = contrast_enhancer.enhance(0.7)
    if save_dir:
        increased_contrast.save(os.path.join(save_dir, f"{file_prefix}_contrast_high.jpg"))
        decreased_contrast.save(os.path.join(save_dir, f"{file_prefix}_contrast_low.jpg"))
        augmented_paths.extend([
            os.path.join(save_dir, f"{file_prefix}_contrast_high.jpg"),
            os.path.join(save_dir, f"{file_prefix}_contrast_low.jpg")
        ])
    augmented_images.extend([np.array(increased_contrast) / 255.0, np.array(decreased_contrast) / 255.0])
    
    # 5. Combine some techniques (rotation + flip)
    combined = ImageOps.mirror(img_pil.rotate(90))
    if save_dir:
        combined.save(os.path.join(save_dir, f"{file_prefix}_rot90_flip.jpg"))
        augmented_paths.append(os.path.join(save_dir, f"{file_prefix}_rot90_flip.jpg"))
    augmented_images.append(np.array(combined) / 255.0)
    
    # 6. Add slight Gaussian noise
    noisy_img = img_pil.copy()
    noisy_array = np.array(noisy_img)
    noise = np.random.normal(0, 15, noisy_array.shape).astype(np.uint8)
    noisy_array = np.clip(noisy_array + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_array)
    if save_dir:
        noisy_img.save(os.path.join(save_dir, f"{file_prefix}_noise.jpg"))
        augmented_paths.append(os.path.join(save_dir, f"{file_prefix}_noise.jpg"))
    augmented_images.append(np.array(noisy_img) / 255.0)
    
    # Add advanced augmentations if all_images is provided
    if all_images is not None and len(all_images) > 1:
        # Select a random image for mixing (not the same as current image)
        img_array = np.array(img_pil)
        
        # Find an image that's not the same as current
        random_idx = random.randint(0, len(all_images) - 1)
        random_img = all_images[random_idx]
        
        # 7. Mixup
        mixup_img = mixup_images(img, random_img, alpha=0.2)
        if save_dir:
            mixup_img_pil = Image.fromarray((mixup_img * 255).astype(np.uint8))
            mixup_img_pil.save(os.path.join(save_dir, f"{file_prefix}_mixup.jpg"))
            augmented_paths.append(os.path.join(save_dir, f"{file_prefix}_mixup.jpg"))
        augmented_images.append(mixup_img)
        
        # 8. CutMix
        cutmix_img = cutmix_images(img, random_img)
        if save_dir:
            cutmix_img_pil = Image.fromarray((cutmix_img * 255).astype(np.uint8))
            cutmix_img_pil.save(os.path.join(save_dir, f"{file_prefix}_cutmix.jpg"))
            augmented_paths.append(os.path.join(save_dir, f"{file_prefix}_cutmix.jpg"))
        augmented_images.append(cutmix_img)
        
        # 9. Elastic deformation
        elastic_img = elastic_transform(img_array)
        if save_dir:
            elastic_img_pil = Image.fromarray(elastic_img)
            elastic_img_pil.save(os.path.join(save_dir, f"{file_prefix}_elastic.jpg"))
            augmented_paths.append(os.path.join(save_dir, f"{file_prefix}_elastic.jpg"))
        augmented_images.append(np.array(elastic_img) / 255.0)
    
    return augmented_images, augmented_paths

# Function to load and preprocess all training data with enhanced augmentation
def load_and_augment_data_enhanced(labels_df, augment=True):
    X = []
    y = []
    all_images = []  # Store all images for advanced augmentation techniques
    
    # Create a mapping from image id to class label
    id_to_label = dict(zip(labels_df['id'], labels_df['type']))
    
    # Process each folder in the training directory
    folders = ['bao ngu xam trang', 'dui ga baby', 'linh chi trang', 'nam mo']
    folder_to_class = {
        'bao ngu xam trang': 1,  # Nấm bào ngư
        'dui ga baby': 2,        # Nấm đùi gà
        'linh chi trang': 3,     # Nấm linh chi trắng
        'nam mo': 0              # Nấm mỡ
    }
    
    # First pass: Load all original images
    for folder in folders:
        folder_path = os.path.join(TRAIN_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist")
            continue
            
        print(f"Processing folder: {folder}")
        images = glob.glob(os.path.join(folder_path, "*.jpg"))
        
        for img_path in tqdm(images):
            img_id = os.path.basename(img_path).split('.')[0]  # Get image ID without extension
            
            # Skip if image ID is not in labels
            if img_id not in id_to_label:
                print(f"Warning: Image ID {img_id} not found in labels CSV")
                continue
                
            img = preprocess_image(img_path)
            if img is None:
                continue
                
            # Add original image
            X.append(img)
            y.append(id_to_label[img_id])
            all_images.append(img)  # Store for later use in advanced augmentations
    
    # Second pass: Apply augmentations with access to all images
    if augment:
        print("Applying enhanced augmentations...")
        for i, (img, label) in enumerate(tqdm(zip(all_images, list(y)), total=len(all_images))):
            # Find an ID for this class label 
            # (this approach avoids index errors when looking up IDs)
            matching_ids = [id for id, lbl in id_to_label.items() if lbl == label]
            if matching_ids:
                img_id = matching_ids[0]
                # Apply enhanced augmentations
                aug_imgs, _ = augment_image_enhanced(img, img_id, all_images=all_images)
                X.extend(aug_imgs)
                y.extend([label] * len(aug_imgs))
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Convert labels to categorical
    y = to_categorical(y, num_classes=NUM_CLASSES)
    
    print(f"Final dataset shape: X: {X.shape}, y: {y.shape}")
    return X, y

# Define the Patch Creation Layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
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

# Define the Patch Encoding Layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# Build the Vision Transformer Model
def build_vit_model():
    # Calculate input shape and other derived parameters
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    num_patches = (IMG_SIZE // PATCH_SIZE) * (IMG_SIZE // PATCH_SIZE)
    
    # Use the predefined PROJECTION_DIM constant
    projection_dim = PROJECTION_DIM

    # Create model
    inputs = layers.Input(shape=input_shape)
    
    # Create patches
    patches = Patches(PATCH_SIZE)(inputs)
    
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create Transformer blocks
    for _ in range(TRANSFORMER_LAYERS):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=projection_dim // NUM_HEADS, dropout=DROPOUT_RATE
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP with fixed dimensions to match the projection dimension
        x4 = layers.Dense(MLP_UNITS[0], activation=tf.nn.gelu)(x3)
        x4 = layers.Dropout(DROPOUT_RATE)(x4)
        x4 = layers.Dense(MLP_UNITS[1], activation=tf.nn.gelu)(x4)
        x4 = layers.Dropout(DROPOUT_RATE)(x4)
        
        # Skip connection 2
        encoded_patches = layers.Add()([x4, x2])

    # Layer normalization and Global Average Pooling
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    
    # Add MLP for classification
    representation = layers.Dropout(0.3)(representation)
    features = layers.Dense(256, activation="relu")(representation)
    features = layers.Dropout(0.3)(features)
    features = layers.Dense(128, activation="relu")(features)
    
    # Final classification layer
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(features)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs, name="ViT_Mushroom_Classifier")
    
    return model

# Function to train the model
def train_model(X_train, y_train, X_val, y_val):
    # Start timing
    start_time = time.time()
    
    # Build model
    model = build_vit_model()
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Define callbacks for training
    checkpoint = ModelCheckpoint(
        "mushroom_vit_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Save the model
    model.save("mushroom_vit_model_final.h5")
    
    # End timing
    training_time = time.time() - start_time
    print(f"\nTotal training time: {format_time(training_time)}")
    
    return model, history

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, history=None):
    # Start timing
    start_time = time.time()
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate and print classification report
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Nấm mỡ (0)', 'Nấm bào ngư (1)', 'Nấm đùi gà (2)', 'Nấm linh chi trắng (3)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Plot training history if provided
    if history is not None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    # End timing
    evaluation_time = time.time() - start_time
    print(f"\nEvaluation time: {format_time(evaluation_time)}")
    
    return y_pred_classes

# Function to predict on test dataset and create submission file
def predict_test_dataset(model):
    # Start timing
    start_time = time.time()
    
    # Get all test images
    test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    test_ids = [os.path.basename(img).split('.')[0] for img in test_images]
    
    # Preprocess test images
    X_test_submit = []
    for img_path in tqdm(test_images, desc="Processing test images"):
        img = preprocess_image(img_path)
        if img is not None:
            X_test_submit.append(img)
    
    X_test_submit = np.array(X_test_submit)
    
    # Make predictions
    predictions = model.predict(X_test_submit, verbose=1)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_ids,
        'type': pred_classes
    })
    
    # Save to CSV
    submission_df.to_csv('mushroom_predictions.csv', index=False)
    
    # End timing
    prediction_time = time.time() - start_time
    num_images = len(test_images)
    
    print(f"Saved predictions to mushroom_predictions.csv")
    print(f"\nPrediction time for {num_images} images: {format_time(prediction_time)}")
    print(f"Average time per image: {format_time(prediction_time/num_images)}")
    print(f"Throughput: {num_images/prediction_time:.2f} images/second")
    
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
    
    return submission_df

# Main execution flow
if __name__ == "__main__":
    # Measure total execution time
    total_start_time = time.time()
    
    # Load labels
    labels_df = load_labels()
    
    # Load and augment training data
    data_start_time = time.time()
    X, y = load_and_augment_data_enhanced(labels_df, augment=True)
    data_time = time.time() - data_start_time
    print(f"Time to load and augment data: {format_time(data_time)}")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    
    # Train the Vision Transformer model
    print("\n=== Training Vision Transformer Model ===")
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate the model
    print("\n=== Evaluating Vision Transformer Model ===")
    evaluate_model(model, X_val, y_val, history)
    
    # Make predictions on test dataset
    print("\n=== Making Predictions on Test Dataset ===")
    predict_test_dataset(model)
    
    # End timing for the entire process
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {format_time(total_time)}")
    print("Complete! Model saved as 'mushroom_vit_model_final.h5'")