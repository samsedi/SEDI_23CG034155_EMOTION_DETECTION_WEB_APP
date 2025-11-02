"""
Emotion Detection Model Training Script
Run this in Google Colab to train your emotion detection model

Instructions:
1. Upload this file to Google Colab
2. Run all cells in order
3. Upload kaggle.json when prompted
4. Wait for training to complete
5. Download emotion_model.h5
"""

# ============================================================
# STEP 1: Install Required Packages
# ============================================================
print("Installing packages...")
# !pip install kaggle tensorflow opencv-python matplotlib seaborn scikit-learn -q
print("✓ Packages installed\n")

# ============================================================
# STEP 2: Import Libraries
# ============================================================
print("Importing libraries...")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("✓ Libraries imported\n")

# ============================================================
# STEP 3: Setup Kaggle API
# ============================================================
print("Setting up Kaggle API...")
from google.colab import files

print("Please upload your kaggle.json file:")
uploaded = files.upload()

# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
print("✓ Kaggle configured\n")

# ============================================================
# STEP 4: Download FER2013 Dataset
# ============================================================
print("Downloading FER2013 dataset...")
# !kaggle datasets download -d msambare/fer2013
# !unzip -q fer2013.zip
print("✓ Dataset downloaded\n")

# ============================================================
# STEP 5: Load Data
# ============================================================
print("Loading data...")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_data(data_dir):
    images = []
    labels = []

    for idx, emotion in enumerate(emotion_labels):
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue

        print(f"  {emotion}...", end=" ")
        count = 0
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(idx)
                count += 1
        print(f"{count}")

    return np.array(images), np.array(labels)

print("Loading training data:")
X_train, y_train = load_data('train')

print("\nLoading test data:")
X_test, y_test = load_data('test')

print(f"\n✓ Training: {X_train.shape[0]} images")
print(f"✓ Test: {X_test.shape[0]} images\n")

# ============================================================
# STEP 6: Preprocess Data
# ============================================================
print("Preprocessing data...")
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

y_train_cat = to_categorical(y_train, num_classes=7)
y_test_cat = to_categorical(y_test, num_classes=7)
print("✓ Data preprocessed\n")

# ============================================================
# STEP 7: Visualize Samples
# ============================================================
print("Visualizing samples...")
fig, axes = plt.subplots(2, 7, figsize=(15, 5))
for i in range(7):
    idx1 = np.where(y_train == i)[0][0]
    idx2 = np.where(y_train == i)[0][1]

    axes[0, i].imshow(X_train[idx1].squeeze(), cmap='gray')
    axes[0, i].set_title(emotion_labels[i])
    axes[0, i].axis('off')

    axes[1, i].imshow(X_train[idx2].squeeze(), cmap='gray')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.bar(emotion_labels, np.bincount(y_train))
plt.title('Training Data Distribution')
plt.xticks(rotation=45)
plt.show()
print("✓ Visualization complete\n")

# ============================================================
# STEP 8: Build Model
# ============================================================
print("Building model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("✓ Model built\n")
model.summary()

# ============================================================
# STEP 9: Setup Data Augmentation
# ============================================================
print("\nSetting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)
datagen.fit(X_train)
print("✓ Augmentation configured\n")

# ============================================================
# STEP 10: Setup Callbacks
# ============================================================
print("Setting up callbacks...")
callbacks = [
    ModelCheckpoint('best_emotion_model.h5', monitor='val_accuracy',
                   save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=15,
                 restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=5, min_lr=1e-7, verbose=1)
]
print("✓ Callbacks configured\n")

# ============================================================
# STEP 11: TRAIN MODEL (This takes 45-60 minutes)
# ============================================================
print("="*60)
print("STARTING TRAINING")
print("="*60)
print("This will take 45-60 minutes. Get some coffee! ☕")
print("="*60 + "\n")

batch_size = 64
epochs = 50

history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=batch_size),
    validation_data=(X_test, y_test_cat),
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60 + "\n")

# ============================================================
# STEP 12: Evaluate Model
# ============================================================
print("Evaluating model...")
model = tf.keras.models.load_model('best_emotion_model.h5')
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"✓ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"✓ Test Loss: {test_loss:.4f}\n")

# ============================================================
# STEP 13: Plot Training History
# ============================================================
print("Plotting training history...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
print("✓ History plotted\n")

# ============================================================
# STEP 14: Confusion Matrix
# ============================================================
print("Generating confusion matrix...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_labels,
            yticklabels=emotion_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=emotion_labels))
print("✓ Evaluation complete\n")

# ============================================================
# STEP 15: Visualize Predictions
# ============================================================
print("Visualizing predictions...")
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
axes = axes.ravel()

for i in range(15):
    idx = np.random.randint(0, len(X_test))
    axes[i].imshow(X_test[idx].squeeze(), cmap='gray')

    pred = model.predict(X_test[idx:idx+1], verbose=0)
    pred_emotion = emotion_labels[np.argmax(pred)]
    true_emotion = emotion_labels[y_test[idx]]

    color = 'green' if pred_emotion == true_emotion else 'red'
    axes[i].set_title(f'P: {pred_emotion}\nT: {true_emotion}', color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
print("✓ Predictions visualized\n")

# ============================================================
# STEP 16: Save and Download Model
# ============================================================
print("Saving final model...")
model.save('emotion_model.h5')
print("✓ Model saved as 'emotion_model.h5'\n")

print("Downloading model...")
files.download('emotion_model.h5')

print("\n" + "="*60)
print("🎉 ALL DONE! 🎉")
print("="*60)
print("Next steps:")
print("1. Save 'emotion_model.h5' to your project folder")
print("2. Test locally: streamlit run app.py")
print("3. Deploy to Render")
print("="*60)