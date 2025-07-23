import os
import sys
import numpy as np
from PIL import Image as PILImage

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.FacialEmotionRecognition.src.utils.response import build_response
from capsules.FacialEmotionRecognition.src.models.PackageModel import PackageModel, Detection


class FacialEmotionRecognition(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.device = self.request.get_param("ConfigDevice")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

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

        if type(face_img) == str:
            return face_img

        select_face = self.select_face_from_image(image, face_img)
        select_face_pil = PILImage.fromarray(select_face.astype(np.uint8))
        gray = select_face_pil.convert("L")

        img = gray.resize((64, 64))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        bbox = detection[0]["boundingBox"]
        roi_gray = gray.resize((48, 48))
        roi_gray = np.array(roi_gray, dtype=np.float32) / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)

        predicted_emotion = self.model.predict(roi_gray)
        max_index = int(np.argmax(predicted_emotion))
        emotion = emotion_labels[max_index]
        detect = Detection(
            boundingBox=bbox, confidence=predicted_emotion[0][max_index],
            classLabel=emotion, classId=max_index, imgUID=img_uid)
        detection_list.append(detect)
        return detection_list

    def run(self):
        self.prediction = []
        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)

        #  YENİ: Detections kontrolü ekle
        if not hasattr(self.image, 'detections') or self.image.detections is None or len(self.image.detections) == 0:
            # Hata durumunda boş liste döndür ve uyarı ver
            print("  WARNING: No face detections found. Please run Face Detection first.")
            self.prediction = []
            # Boş detection listesi ile devam et
            empty_detection = Detection(
                boundingBox={"left": 0, "top": 0, "width": 0, "height": 0},
                confidence=0.0,
                classLabel="No Face Detected",
                classId=-1,
                imgUID=self.image.uID if hasattr(self.image, 'uID') else "unknown"
            )
            self.prediction = [empty_detection]
        else:
            # Normal akış
            self.prediction = self.infer(self.image.value, self.image.detections, self.image.uID)

        self.image = Image.set_frame(img=self.image, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
