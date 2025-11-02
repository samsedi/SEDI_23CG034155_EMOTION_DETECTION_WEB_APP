"""
Emotion Detection Model
"""

import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model, Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import warnings

warnings.filterwarnings("ignore")


class EmotionDetector:
    """Detect emotions from facial expressions in images."""

    def __init__(self, model_path="emotion_model.h5"):
        """Initialize the emotion detector."""
        self.emotion_labels = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ]

        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Load or create model
        if os.path.exists(model_path):
            self.model = load_model(model_path, compile=False)
            self.model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
        else:
            self.model = self._create_default_model()

    def _create_default_model(self):
        """Create default CNN model."""
        model = Sequential(
            [
                Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=(48, 48, 1),
                    padding="same",
                ),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                Conv2D(64, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                Conv2D(128, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                Conv2D(128, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                Conv2D(256, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                Flatten(),
                Dense(256, activation="relu"),
                BatchNormalization(),
                Dropout(0.5),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(0.5),
                Dense(7, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    def detect_faces(self, image):
        """Detect faces in image."""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        )
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces, gray

    def predict_emotion(self, face_img):
        """Predict emotion from face image."""
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)

        predictions = self.model.predict(face_img, verbose=0)[0]
        emotion_dict = {
            self.emotion_labels[i]: float(predictions[i])
            for i in range(len(self.emotion_labels))
        }
        dominant_emotion = self.emotion_labels[np.argmax(predictions)]

        return emotion_dict, dominant_emotion

    def detect_emotions(self, image):
        """Detect and analyze emotions for all faces in image."""
        output_image = image.copy()
        faces, gray = self.detect_faces(image)
        emotions_list = []

        for idx, (x, y, w, h) in enumerate(faces):
            face_roi = gray[y : y + h, x : x + w]
            emotion_dict, dominant_emotion = self.predict_emotion(face_roi)

            emotions_list.append(
                {
                    "emotions": emotion_dict,
                    "dominant_emotion": dominant_emotion,
                    "box": (x, y, w, h),
                }
            )

            # Draw rectangle and label
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{dominant_emotion.upper()}: {emotion_dict[dominant_emotion]:.1%}"
            cv2.putText(
                output_image,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return output_image, emotions_list


def train_emotion_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    """Train emotion detection model."""
    model = Sequential(
        [
            Conv2D(
                32, (3, 3), activation="relu", input_shape=(48, 48, 1), padding="same"
            ),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(7, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(
            "emotion_model.h5",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history
