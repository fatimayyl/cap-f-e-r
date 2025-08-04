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
        self.device = self.request.get_param("ConfigDevice")
        self.select_device = self.bootstrap["device"]
        if self.device == "GPU" and "GPU" in self.select_device:
            self.model = self.bootstrap["ModelGPU"]["model"]
        else:
            self.model = self.bootstrap["ModelCPU"]["model"]

        print("Model initialized:", self.model)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model = load_models()
        print("Model loaded in bootstrap:", model)
        return model

    def filter_bbox_face(self, face_detect):
        if len(face_detect) == 0:
            return "Face information not found"
        if len(face_detect) > 1:
            return "Multiple face information found"
        if len(face_detect) == 1:
            return face_detect[0].boundingBox  # Detection objesi

    def select_face_from_image(self, image, bbox):
        left = int(bbox["left"])
        top = int(bbox["top"])
        width = int(bbox["width"])
        height = int(bbox["height"])
        face_image = image[top:top + height, left:left + width]
        return face_image

    def infer(self, image, detection, img_uid):
        detection_list = []
        if not detection or len(detection) == 0:
            return detection_list  # boş liste döndür

        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        face_img = self.filter_bbox_face(detection)
        if isinstance(face_img, str):
            # Hata mesajı stringi yerine boş liste dön
            return detection_list

        select_face = self.select_face_from_image(image, face_img)
        select_face_pil = PILImage.fromarray(select_face.astype(np.uint8))
        gray = select_face_pil.convert("L")

        # Resize and normalize
        img = gray.resize((64, 64))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        bbox = detection[0].boundingBox  # Detection objesi

        roi_gray = gray.resize((48, 48))
        roi_gray = np.array(roi_gray, dtype=np.float32) / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)

        predicted_emotion = self.model.predict(roi_gray)
        max_index = int(np.argmax(predicted_emotion))
        emotion = emotion_labels[max_index]

        detect = Detection(
            boundingBox=bbox,
            confidence=float(predicted_emotion[0][max_index]),
            classLabel=emotion,
            classId=max_index,
            imgUID=img_uid
        )
        detection_list.append(detect)
        return detection_list

    def run(self):
        print("DEBUG: full request data:", self.request.data)

        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        print("DEBUG: detections before infer:", getattr(self.image, "detections", None))

        detections_in_request = (
            self.request.data.get("inputs", {})
            .get("inputDetections", {})
            .get("value", [])
        )

        if detections_in_request:
            self.image.detections = [Detection(**det) for det in detections_in_request]
            print("DEBUG: detections set from request.data:", self.image.detections)
        else:
            self.image.detections = []
            print("DEBUG: no detections found in request; skipping infer")

        if not self.image.detections:
            self.prediction = []
        else:
            self.prediction = self.infer(
                self.image.value,
                getattr(self.image, "detections", []),
                self.image.uID
            )
            print("DEBUG: prediction from infer:", self.prediction)

        # ⚡ Pydantic hatasını engellemek için sadece Detection objelerini serialize et
        self.image.detections = self.prediction

        self.image = Image.set_frame(
            img=self.image, package_uID=self.uID, redis_db=self.redis_db
        )

        # PackageModel için context oluştur
        packageModel = build_response(context=self)
        print("DEBUG: final packageModel:", packageModel)
        return packageModel


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
