# Pneumonia Detection with DenseNet121, OpenCV, Augmentation, and Class Balancing
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
from sklearn.utils import shuffle

# =========================
# 1. Parameters
# =========================
DATA_DIR = r"D:\PneumoniaDetector\dataset"
IMG_SIZE = 224       # DenseNet121 expects 224x224
BATCH_SIZE = 16

# =========================
# 2. Load and preprocess images using OpenCV
# =========================
def augment_image(img):
    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    # Random rotation
    angle = np.random.randint(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), angle, 1)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
    return img

def load_data(data_dir, augment=False):
    data = []
    labels = []

    for label, folder in enumerate(['NORMAL', 'PNEUMONIA']):
        path = os.path.join(data_dir, folder)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            if augment and label == 1:  # Only augment pneumonia images (optional)
                img = augment_image(img)
            data.append(img)
            labels.append(label)
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    return data, labels

# Load train, val, test
X_train, y_train = load_data(os.path.join(DATA_DIR, 'train'), augment=True)
X_val, y_val = load_data(os.path.join(DATA_DIR, 'val'))
X_test, y_test = load_data(os.path.join(DATA_DIR, 'test'))

# Shuffle training data
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# =========================
# 3. Class weights for imbalance
# =========================
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights:", class_weights_dict)

# =========================
# 4. Build Transfer Learning Model
# =========================
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# =========================
# 5. Callbacks
# =========================
checkpoint = ModelCheckpoint('pneumonia_densenet.h5', save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# =========================
# 6. Train model
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=BATCH_SIZE,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, early_stop]
)

# =========================
# 7. Evaluate on test set
# =========================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# =========================
# 8. Plot training history
# =========================
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.show()

# =========================
# 9. Predict new image
# =========================
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    label = "PNEUMONIA" if pred[0][0] > 0.5 else "NORMAL"
    print("Prediction:", label)
    return label

# Example:
# predict_image("test_normal.jpg")
