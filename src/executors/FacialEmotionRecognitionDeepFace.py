import os
import sys
import numpy as np
from PIL import Image as PILImage



sys.path.append('/opt/project/capsules/FacialEmotionRecognition/src/lib/deepface')

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))



from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.FacialEmotionRecognition.src.models.PackageModel import PackageModel, Detection, ConfigReturnAllScores
from capsules.FacialEmotionRecognition.src.utils.response import build_response_deepface



# import deepface
from deepface import DeepFace


class FacialEmotionRecognitionDeepFace(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.input_detections = self.request.get_param("inputDetections")
        print("self.input_detections :", self.input_detections )

        # detections'ı Image nesnesine sonradan eklemek için
        if self.input_detections:
            self.image["detections"] = self.input_detections

        self.return_all_scores = self.request.get_param("ReturnAllScores")



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

            emotion = result["dominant_emotion"]
            confidence = result["emotion"][emotion]
            class_id = list(result["emotion"].keys()).index(emotion)

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


        except Exception as e:
            print(f"DeepFace error: {str(e)}")
            return []


        return detection_list

    """
    def run(self):

        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        self.detections = self.deepface_inference()
        packageModel = build_response_deepface(context=self)
        return packageModel
    """

    def run(self):

        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if self.input_detections:
            self.image.detections = self.input_detections
        else:
            self.image.detections = []

        self.detections = self.deepface_inference()
        packageModel = build_response_deepface(context=self)
        return packageModel

if __name__ == "__main__":
    Executor(sys.argv[1]).run()
