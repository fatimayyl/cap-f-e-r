import os
import sys
import numpy as np
from PIL import Image as PILImage



sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
"""
deepface_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../deepface'))
sys.path.append(deepface_path)
"""
from deepface import DeepFace


from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.FacialEmotionRecognition.src.models.PackageModel import PackageModel, Detection #,ReturnAllScores
from capsules.FacialEmotionRecognition.src.utils.response import build_response_deepface


class FacialEmotionRecognitionDeepFace(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.return_all_scores = self.request.get_param("ReturnAllScores")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def deepface_inference(self):
        detection_list = []
        self.image.value = np.asarray(self.image.value).astype(np.uint8)

        # DeepFace ile tespit yapılır (detector_backend opencv)
        try:
            results = DeepFace.analyze(
                img_path=self.image.value,
                actions=["emotion"],
                enforce_detection=True,
                detector_backend="opencv"
            )
        except Exception as e:
            print(f"[❌ DeepFace error] {e}")
            return []

        # Multi-face destekliyoruz ama her biri için Detection nesnesi oluşturulmalı
        for result in results:
            region = result.get("region", {})
            emotion = result["dominant_emotion"]
            confidence = result["emotion"][emotion]
            class_id = list(result["emotion"].keys()).index(emotion)

            bbox = {
                "left": region["x"],
                "top": region["y"],
                "width": region["w"],
                "height": region["h"]
            }

            detection = Detection(
                boundingBox=bbox,
                confidence=confidence,
                classLabel=emotion.capitalize(),
                classId=class_id,
                imgUID=self.image.uID
            )

            if self.return_all_scores:
                detection.extra = {"emotion_scores": result["emotion"]}

            detection_list.append(detection)

        return detection_list

    def run(self):
        # Redis’ten img yükle
        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)

        # Yüz tespiti + duygular
        detection_list = self.deepface_inference()

        # 📌 En önemli satır: image.detections'a yaz!
        self.image.detections = detection_list

        # response’a bağla
        packageModel = build_response_deepface(context=self)
        return packageModel


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
