import os
import sys
import numpy as np
from PIL import Image as PILImage

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.FacialEmotionRecognition.src.utils.utils import load_models
from capsules.FacialEmotionRecognition.src.utils.response import build_response
from capsules.FacialEmotionRecognition.src.models.PackageModel import PackageModel, Detection


class FacialEmotionRecognition(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.device = self.bootstrap.get("ConfigDevice")
        print("self.device:", self.device)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return load_models(config=config)

    def filter_bbox_face(self, face_detect):
        if len(face_detect) == 0:
            return "Face information not found"
        if len(face_detect) > 1:
            return "Multiple face information found"
        if len(face_detect) == 1:
            return face_detect[0]["boundingBox"]

    def select_face_from_image(self, image, bbox):
        left = int(bbox["left"])
        top = int(bbox["top"])
        width = int(bbox["width"])
        height = int(bbox["height"])
        face_image = image[top:top + height, left:left + width]
        return face_image

    def infer(self, image, detection, img_uid):
        detection_list = []
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        face_img = self.filter_bbox_face(detection)

        if isinstance(face_img, str):
            return face_img

        select_face = self.select_face_from_image(image, face_img)
        select_face_pil = PILImage.fromarray(select_face.astype(np.uint8))
        gray = select_face_pil.convert("L")

        # Model input: 48x48, normalize, expand dims (batch, height, width, channel)
        roi_gray = gray.resize((48, 48))
        roi_gray = np.array(roi_gray, dtype=np.float32) / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # kanal ekle
        roi_gray = np.expand_dims(roi_gray, axis=0)   # batch ekle

        predicted_emotion = self.model.predict(roi_gray)
        max_index = int(np.argmax(predicted_emotion))
        emotion = emotion_labels[max_index]
        confidence = float(predicted_emotion[0][max_index])

        bbox = detection[0]["boundingBox"]

        detect = Detection(
            boundingBox=bbox,
            confidence=confidence,
            classLabel=emotion,
            classId=max_index,
            imgUID=img_uid
        )
        detection_list.append(detect)
        return detection_list

    def run(self):
        self.prediction = []
        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if not self.image:
            return None

        # image.detections burada yüz tespit sonuçları olmalı, yoksa yüz tespiti eklenmeli.
        if not hasattr(self.image, "detections") or not self.image.detections:
            print("No face detections found in image.detections!")
            return None

        self.prediction = self.infer(self.image.value, self.image.detections, self.image.uID)
        self.image = Image.set_frame(img=self.image, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response(context=self)
        return packageModel


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
