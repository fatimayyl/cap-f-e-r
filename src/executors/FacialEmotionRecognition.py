import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64

from sdks.novavision.src.base.executor import Executor
from sdks.novavision.src.base.model import Image, OutputImage

from capsules.FacialEmotionRecognition.src.models.PackageModel import (
    FacialEmotionRecognitionOutputs,
    FacialEmotionRecognitionResponse,
    OutputDetections,
    Detection,
    BoundingBox
)


class FacialEmotionRecognitionExecutor(Executor):
    def __init__(self):
        super().__init__()
        self.model = None
        self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def load_model(self):
        weight_path = '/storage/modelFER.h5'
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Model weight not found at {weight_path}")
        self.model = load_model(weight_path)
        print("✅ Model loaded from:", weight_path)

    def normalize_image(self, image):
        if image.dtype != np.uint8:
            print("🛠️ Normalizing image to uint8...")
            image = (255 * image).clip(0, 255).astype(np.uint8)

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (48, 48))
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        return image

    def annotate_image(self, image_np, label):
        annotated = image_np.copy()
        if annotated.ndim == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        return annotated

    def encode_image_to_base64(self, image_np):
        success, encoded_img = cv2.imencode('.jpg', image_np)
        if not success:
            raise ValueError("cv2.imencode failed to encode image.")
        return base64.b64encode(encoded_img.tobytes()).decode('utf-8')

    def run(self):
        if self.model is None:
            self.load_model()

        # Doğru input key
        image: Image = self.inputs.inputImage.value
        image_np = image.value

        print(f"📥 Input image shape: {image_np.shape}, dtype: {image_np.dtype}")

        input_tensor = self.normalize_image(image_np)

        predictions = self.model.predict(input_tensor)
        emotion_index = np.argmax(predictions[0])
        emotion_label = self.labels[emotion_index]
        confidence = float(predictions[0][emotion_index])

        print(f"🔍 Prediction: {emotion_label} ({confidence:.2f})")

        # Annotate image
        annotated_np = self.annotate_image(image_np, f"{emotion_label} ({confidence:.2f})")
        image.bytes = self.encode_image_to_base64(annotated_np)

        # OutputImage
        output_image = OutputImage(name="outputImage", value=image)

        # Optional: dummy detection
        detection = Detection(
            boundingBox=BoundingBox(left=0, top=0, width=1, height=1),
            imgUID=image.uid,
            label=emotion_label,
            score=confidence
        )
        output_detections = OutputDetections(name="outputDetections", value=[detection])

        # Final outputs
        outputs = FacialEmotionRecognitionOutputs(
            outputImage=output_image,
            outputDetections=output_detections
        )
        self.response = FacialEmotionRecognitionResponse(outputs=outputs)
