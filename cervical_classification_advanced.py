# ==================== Advanced Cervical Image Classification ====================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4, DenseNet121, ResNet50V2
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout, 
                                   BatchNormalization, Conv2D, MaxPooling2D,
                                   UpSampling2D, Input, concatenate, Multiply)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, 
                                      ReduceLROnPlateau, LearningRateScheduler)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import os
import random
from scipy import ndimage
import albumentations as A
from tensorflow.keras.utils import Sequence
import tensorflow_addons as tfa

# ==================== Reproducibility Setup ====================
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ==================== Configuration ====================
class Config:
    IMG_SIZE = (512, 512)  # Increased size for better feature extraction
    BATCH_SIZE = 16  # Reduced for larger images
    EPOCHS_INITIAL = 20
    EPOCHS_FINE_TUNE = 15
    LEARNING_RATE_INITIAL = 1e-4
    LEARNING_RATE_FINE_TUNE = 1e-5
    DROPOUT_RATE = 0.3
    L2_REG = 1e-4
    
    # Paths
    BASE_PATH = '/content/drive/MyDrive/LastDatasetVedant'
    TRAIN_DIR = os.path.join(BASE_PATH, 'Train')
    TEST_DIR = os.path.join(BASE_PATH, 'Test')

config = Config()

# ==================== Advanced Data Augmentation ====================
class AdvancedAugmentation:
    def __init__(self):
        self.albumentations_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.3),
        ])
    
    def apply_medical_augmentation(self, image):
        """Apply medical-specific augmentations"""
        if random.random() < 0.3:
            # Simulate different lighting conditions
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() < 0.3:
            # Simulate different contrast settings
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() < 0.2:
            # Add slight blur to simulate focus variations
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        return image

# ==================== Custom Data Generator ====================
class CervicalDataGenerator(Sequence):
    def __init__(self, directory, batch_size, img_size, is_training=True, augment=True):
        self.directory = directory
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_training = is_training
        self.augment = augment
        self.augmenter = AdvancedAugmentation()
        
        # Load file paths and labels
        self.file_paths = []
        self.labels = []
        
        for class_name in ['Normal', 'Abnormal']:
            class_dir = os.path.join(directory, class_name)
            label = 0 if class_name == 'Normal' else 1
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.file_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(label)
        
        self.indexes = np.arange(len(self.file_paths))
        if is_training:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return len(self.file_paths) // self.batch_size
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        X = np.zeros((self.batch_size, *self.img_size, 3), dtype=np.float32)
        y = np.zeros((self.batch_size,), dtype=np.float32)
        
        for i, idx in enumerate(batch_indexes):
            image = self.load_and_preprocess_image(self.file_paths[idx])
            X[i] = image
            y[i] = self.labels[idx]
        
        return X, y
    
    def load_and_preprocess_image(self, path):
        # Load image
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Cervix region detection and cropping
        image = self.detect_and_crop_cervix(image)
        
        # Resize
        image = cv2.resize(image, self.img_size)
        
        # Apply augmentations if training
        if self.is_training and self.augment:
            image = self.augmenter.albumentations_transform(image=image)['image']
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def detect_and_crop_cervix(self, image):
        """Detect and crop cervix region using computer vision techniques"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create mask for pink/red regions (cervix tissue)
        lower_pink = np.array([0, 50, 50])
        upper_pink = np.array([20, 255, 255])
        mask1 = cv2.inRange(hsv, lower_pink, upper_pink)
        
        lower_pink2 = np.array([160, 50, 50])
        upper_pink2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Crop the image
            if w > 50 and h > 50:  # Ensure minimum size
                image = image[y:y+h, x:x+w]
        
        return image
    
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

# ==================== Attention Mechanism ====================
def attention_block(inputs, filters):
    """Channel and Spatial Attention Block"""
    # Channel Attention
    channel_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(inputs)
    channel_max_pool = tf.keras.layers.GlobalMaxPooling2D(keepdims=True)(inputs)
    
    channel_attention = tf.keras.layers.Dense(filters // 8, activation='relu')(channel_avg_pool)
    channel_attention = tf.keras.layers.Dense(filters, activation='sigmoid')(channel_attention)
    
    channel_attention_max = tf.keras.layers.Dense(filters // 8, activation='relu')(channel_max_pool)
    channel_attention_max = tf.keras.layers.Dense(filters, activation='sigmoid')(channel_attention_max)
    
    channel_attention = tf.keras.layers.Add()([channel_attention, channel_attention_max])
    
    # Apply channel attention
    x = tf.keras.layers.Multiply()([inputs, channel_attention])
    
    # Spatial Attention
    spatial_avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    spatial_max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    spatial_concat = tf.keras.layers.Concatenate(axis=-1)([spatial_avg_pool, spatial_max_pool])
    
    spatial_attention = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')(spatial_concat)
    
    # Apply spatial attention
    output = tf.keras.layers.Multiply()([x, spatial_attention])
    
    return output

# ==================== Advanced Model Architecture ====================
def build_advanced_model():
    """Build model with attention mechanisms and multiple feature extractors"""
    inputs = Input(shape=(*config.IMG_SIZE, 3))
    
    # EfficientNet backbone
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=(*config.IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    # Extract features
    x = base_model(inputs, training=False)
    
    # Add attention mechanism
    x = attention_block(x, x.shape[-1])
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers with regularization
    x = Dense(512, activation='relu', 
              kernel_regularizer=l2(config.L2_REG),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    
    x = Dense(256, activation='relu',
              kernel_regularizer=l2(config.L2_REG),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    
    x = Dense(128, activation='relu',
              kernel_regularizer=l2(config.L2_REG),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid', name='classification')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model

# ==================== Ensemble Model ====================
def build_ensemble_models():
    """Build multiple models for ensemble"""
    models = []
    
    # Model 1: EfficientNetB4
    inputs = Input(shape=(*config.IMG_SIZE, 3))
    base1 = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(*config.IMG_SIZE, 3))
    base1.trainable = False
    x1 = base1(inputs, training=False)
    x1 = attention_block(x1, x1.shape[-1])
    x1 = GlobalAveragePooling2D()(x1)
    x1 = Dense(256, activation='relu', kernel_regularizer=l2(config.L2_REG))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(config.DROPOUT_RATE)(x1)
    out1 = Dense(1, activation='sigmoid')(x1)
    model1 = Model(inputs, out1)
    models.append(('EfficientNetB4', model1, base1))
    
    # Model 2: DenseNet121
    base2 = DenseNet121(weights='imagenet', include_top=False, input_shape=(*config.IMG_SIZE, 3))
    base2.trainable = False
    x2 = base2(inputs, training=False)
    x2 = attention_block(x2, x2.shape[-1])
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dense(256, activation='relu', kernel_regularizer=l2(config.L2_REG))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(config.DROPOUT_RATE)(x2)
    out2 = Dense(1, activation='sigmoid')(x2)
    model2 = Model(inputs, out2)
    models.append(('DenseNet121', model2, base2))
    
    # Model 3: ResNet50V2
    base3 = ResNet50V2(weights='imagenet', include_top=False, input_shape=(*config.IMG_SIZE, 3))
    base3.trainable = False
    x3 = base3(inputs, training=False)
    x3 = attention_block(x3, x3.shape[-1])
    x3 = GlobalAveragePooling2D()(x3)
    x3 = Dense(256, activation='relu', kernel_regularizer=l2(config.L2_REG))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(config.DROPOUT_RATE)(x3)
    out3 = Dense(1, activation='sigmoid')(x3)
    model3 = Model(inputs, out3)
    models.append(('ResNet50V2', model3, base3))
    
    return models

# ==================== Advanced Training ====================
def get_callbacks(model_name):
    """Get training callbacks"""
    return [
        ModelCheckpoint(
            f'best_{model_name}_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

def compile_model(model, learning_rate):
    """Compile model with advanced metrics"""
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tfa.metrics.F1Score(num_classes=1, threshold=0.5, name='f1_score')
        ]
    )
    return model

# ==================== Training Pipeline ====================
def train_model(model, base_model, model_name, train_gen, val_gen):
    """Complete training pipeline"""
    print(f"\n=== Training {model_name} ===")
    
    # Compile model
    model = compile_model(model, config.LEARNING_RATE_INITIAL)
    
    # Calculate class weights
    train_labels = []
    for i in range(len(train_gen)):
        _, labels = train_gen[i]
        train_labels.extend(labels)
    
    class_counts = np.bincount(np.array(train_labels).astype(int))
    total_samples = len(train_labels)
    class_weight = {
        0: total_samples / (2 * class_counts[0]),
        1: total_samples / (2 * class_counts[1])
    }
    
    print(f"Class weights: {class_weight}")
    
    # Initial training
    callbacks = get_callbacks(model_name)
    
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_INITIAL,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # Fine-tuning
    print(f"\n=== Fine-tuning {model_name} ===")
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model = compile_model(model, config.LEARNING_RATE_FINE_TUNE)
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_FINE_TUNE,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # Save final model
    model.save(f'final_{model_name}_model.h5')
    
    return model, history1, history2

# ==================== Evaluation Functions ====================
def evaluate_model(model, generator, set_name, model_name):
    """Comprehensive model evaluation"""
    print(f"\n=== {set_name} Set Evaluation for {model_name} ===")
    
    # Get predictions
    y_true = []
    y_pred = []
    
    for i in range(len(generator)):
        X_batch, y_batch = generator[i]
        pred_batch = model.predict(X_batch, verbose=0)
        y_true.extend(y_batch)
        y_pred.extend(pred_batch.flatten())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_class = (y_pred > 0.5).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_class, 
                              target_names=['Normal', 'Abnormal']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title(f'Confusion Matrix - {set_name} ({model_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {set_name} ({model_name})')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    return roc_auc, y_pred

# ==================== Ensemble Prediction ====================
def ensemble_predict(models, X):
    """Ensemble prediction using multiple models"""
    predictions = []
    for _, model, _ in models:
        pred = model.predict(X, verbose=0)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

# ==================== Main Training Pipeline ====================
def main():
    print("=== Advanced Cervical Image Classification ===")
    
    # Create data generators
    print("Creating data generators...")
    train_gen = CervicalDataGenerator(
        config.TRAIN_DIR, 
        config.BATCH_SIZE, 
        config.IMG_SIZE, 
        is_training=True, 
        augment=True
    )
    
    # Split training data for validation
    train_size = int(0.8 * len(train_gen.file_paths))
    val_paths = train_gen.file_paths[train_size:]
    val_labels = train_gen.labels[train_size:]
    train_gen.file_paths = train_gen.file_paths[:train_size]
    train_gen.labels = train_gen.labels[:train_size]
    
    # Create validation generator
    val_gen = CervicalDataGenerator(
        config.TRAIN_DIR,
        config.BATCH_SIZE,
        config.IMG_SIZE,
        is_training=False,
        augment=False
    )
    val_gen.file_paths = val_paths
    val_gen.labels = val_labels
    
    # Create test generator
    test_gen = CervicalDataGenerator(
        config.TEST_DIR,
        config.BATCH_SIZE,
        config.IMG_SIZE,
        is_training=False,
        augment=False
    )
    
    print(f"Training samples: {len(train_gen.file_paths)}")
    print(f"Validation samples: {len(val_gen.file_paths)}")
    print(f"Test samples: {len(test_gen.file_paths)}")
    
    # Build and train ensemble models
    ensemble_models = build_ensemble_models()
    trained_models = []
    
    for model_name, model, base_model in ensemble_models:
        trained_model, hist1, hist2 = train_model(
            model, base_model, model_name, train_gen, val_gen
        )
        trained_models.append((model_name, trained_model, base_model))
        
        # Evaluate individual model
        val_auc, _ = evaluate_model(trained_model, val_gen, "Validation", model_name)
        test_auc, _ = evaluate_model(trained_model, test_gen, "Test", model_name)
        
        print(f"{model_name} - Validation AUC: {val_auc:.3f}, Test AUC: {test_auc:.3f}")
    
    # Ensemble evaluation
    print("\n=== Ensemble Model Evaluation ===")
    
    # Validation ensemble
    val_ensemble_pred = []
    for i in range(len(val_gen)):
        X_batch, _ = val_gen[i]
        ensemble_pred = ensemble_predict(trained_models, X_batch)
        val_ensemble_pred.extend(ensemble_pred.flatten())
    
    val_true = []
    for i in range(len(val_gen)):
        _, y_batch = val_gen[i]
        val_true.extend(y_batch)
    
    val_ensemble_pred = np.array(val_ensemble_pred)
    val_true = np.array(val_true)
    val_pred_class = (val_ensemble_pred > 0.5).astype(int)
    
    print("\nEnsemble Validation Results:")
    print(classification_report(val_true, val_pred_class, 
                              target_names=['Normal', 'Abnormal']))
    
    # Test ensemble
    test_ensemble_pred = []
    for i in range(len(test_gen)):
        X_batch, _ = test_gen[i]
        ensemble_pred = ensemble_predict(trained_models, X_batch)
        test_ensemble_pred.extend(ensemble_pred.flatten())
    
    test_true = []
    for i in range(len(test_gen)):
        _, y_batch = test_gen[i]
        test_true.extend(y_batch)
    
    test_ensemble_pred = np.array(test_ensemble_pred)
    test_true = np.array(test_true)
    test_pred_class = (test_ensemble_pred > 0.5).astype(int)
    
    print("\nEnsemble Test Results:")
    print(classification_report(test_true, test_pred_class,
                              target_names=['Normal', 'Abnormal']))
    
    # Final ROC curves comparison
    plt.figure(figsize=(12, 8))
    
    for model_name, model, _ in trained_models:
        y_pred = []
        for i in range(len(test_gen)):
            X_batch, _ = test_gen[i]
            pred_batch = model.predict(X_batch, verbose=0)
            y_pred.extend(pred_batch.flatten())
        
        fpr, tpr, _ = roc_curve(test_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Ensemble ROC
    fpr, tpr, _ = roc_curve(test_true, test_ensemble_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=3, linestyle='--', 
             label=f'Ensemble (AUC = {roc_auc:.3f})', color='red')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison - Test Set')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    print(f"\nFinal Ensemble Test AUC: {roc_auc:.3f}")
    
    return trained_models

if __name__ == "__main__":
    # Mount drive first
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Run main training
    models = main()