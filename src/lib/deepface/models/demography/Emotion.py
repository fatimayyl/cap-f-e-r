# built-in dependencies
import os

# 3rd party dependencies
import gdown
import numpy as np
import cv2
import requests

# project dependencies
from deepface.commons import package_utils, folder_utils
from deepface.models.Demography import Demography
from deepface.commons.logger import Logger


# -------------------------------------------
# pylint: disable=line-too-long
# -------------------------------------------
# dependency configuration
tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
else:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D,
        MaxPooling2D,
        AveragePooling2D,
        Flatten,
        Dense,
        Dropout,
    )
# -------------------------------------------

# Labels for the emotions that can be detected by the model.
labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# pylint: disable=too-few-public-methods
class EmotionClient(Demography):


    def __init__(self):
        self.model = load_model()
        self.model_name = "Emotion"

    def predict(self, img: np.ndarray) -> np.ndarray:
        img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        img_gray = np.expand_dims(img_gray, axis=0)

        # model.predict causes memory issue when it is called in a for loop
        # emotion_predictions = self.model.predict(img_gray, verbose=0)[0, :]
        emotion_predictions = self.model(img_gray, training=False).numpy()[0, :]

        return emotion_predictions

def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5",
) -> Sequential:

    num_classes = 7
    model = Sequential()

    # --- Katmanlar ---
    model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))

    # --- Model dosyası yolu ---
    output = "/storage/facial_expression_model_weights.h5"

    if not os.path.isfile(output):
        print(f"Model file not found at {output}, downloading...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    model.load_weights(output)

    return model







