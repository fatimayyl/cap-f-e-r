import os
import sys
import numpy as np
from PIL import Image as PILImage
from deepface import DeepFace

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.FacialEmotionRecognition.src.models.PackageModel import PackageModel, Detection
from capsules.FacialEmotionRecognition.src.utils.response import build_response_deepface


class FacialEmotionRecognitionDeepFace(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def deepface_inference(self):
        detection_list = []
        self.image.value = np.asarray(self.image.value).astype(np.uint8)

        if len(self.image.detections) == 0:
            return "Face information not found"
        if len(self.image.detections) > 1:
            return "Multiple face information found"

        bbox = self.image.detections[0]["boundingBox"]
        x, y, w, h = int(bbox["left"]), int(bbox["top"]), int(bbox["width"]), int(bbox["height"])
        face_crop = self.image.value[y:y+h, x:x+w]

        try:
            pil_image = PILImage.fromarray(face_crop.astype(np.uint8))
            result = DeepFace.analyze(
                img_path=np.array(pil_image),
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv"
            )[0]

            emotion_scores = result["emotion"]
            for idx, (label, score) in enumerate(emotion_scores.items()):
                detection = Detection(
                    boundingBox=bbox,
                    confidence=score,
                    classLabel=label.capitalize(),
                    classId=idx,
                    imgUID=self.image.uID
                )
                detection_list.append(detection)

        except Exception as e:
            return f"DeepFace error: {str(e)}"

        return detection_list

    def run(self):
        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        self.detections = self.deepface_inference()
        packageModel = build_response_deepface(context=self)
        return packageModel


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
