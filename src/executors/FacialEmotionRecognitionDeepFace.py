import os
import sys
import numpy as np
from PIL import Image as PILImage

sys.path.append('/opt/project/capsules/FacialEmotionRecognition/src/lib')
from deepface.DeepFace import analyze, verify, build_model

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.FacialEmotionRecognition.src.utils.utilsdeepface import load_emotion_model
from capsules.FacialEmotionRecognition.src.models.PackageModel import PackageModel, Detection, ConfigReturnAllScores
from capsules.FacialEmotionRecognition.src.utils.response import build_response_deepface


class FacialEmotionRecognitionDeepFace(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        
        # Config'den returnAllScores değerini al
        try:
            config_return_all_scores = self.request.get_param("configReturnAllScores")
            if config_return_all_scores and hasattr(config_return_all_scores, 'value'):
                self.return_all_scores = config_return_all_scores.value == "True"
            else:
                self.return_all_scores = False
        except Exception as e:
            print(f"Config error: {e}")
            self.return_all_scores = False
            
        print(f"Return all scores setting: {self.return_all_scores}")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model = load_emotion_model(config)
        print("Model loaded in bootstrap:", model)
        return {
            "device": "/device:CPU:0",
            "ModelCPU": {"model": model}
        }

    def deepface_inference(self):
        detection_list = []
        
        # Denenmesi gereken detector'lar sırasıyla
        detectors = ["opencv", "mtcnn", "retinaface", "ssd"]
        
        try:
            # DeepFace'e numpy array olarak geçiriyoruz
            self.image.value = np.asarray(self.image.value).astype(np.uint8)
            
            result = None
            successful_detector = None
            
            # Farklı detector'ları sırasıyla dene
            for detector in detectors:
                try:
                    print(f"Trying detector: {detector}")
                    result = analyze(
                        img_path=self.image.value,
                        actions=["emotion"],
                        enforce_detection=False,  # Yüz algılanamasa bile devam et
                        detector_backend=detector,
                        silent=True
                    )
                    successful_detector = detector
                    print(f"Success with detector: {detector}")
                    break
                except Exception as detector_error:
                    print(f"Detector {detector} failed: {str(detector_error)}")
                    continue
            
            # Hiçbir detector başarılı olamadıysa
            if result is None:
                print("No detector could analyze the image")
                return []
            
            # Eğer birden fazla yüz bulunduysa, hepsini işle
            if isinstance(result, list):
                faces = result
            else:
                faces = [result]
            
            for face_result in faces:
                # Yüz bounding box bilgisini al
                region = face_result.get("region", {})
                
                if not region:  # Region bilgisi yoksa, tüm resmi kullan
                    height, width = self.image.value.shape[:2]
                    region = {"x": 0, "y": 0, "w": width, "h": height}
                
                # Bounding box koordinatlarını çıkar
                x = region.get("x", 0)
                y = region.get("y", 0)
                w = region.get("w", 0)
                h = region.get("h", 0)
                
                # Bounding box dict'ini oluştur
                bbox = {
                    "left": x,
                    "top": y,
                    "width": w,
                    "height": h
                }
                
                # Duygu analizi sonuçlarını al
                emotion = face_result["dominant_emotion"]
                confidence = face_result["emotion"][emotion]
                class_id = list(face_result["emotion"].keys()).index(emotion)
                
                # Detection objesi oluştur
                detection = Detection(
                    boundingBox=bbox,
                    confidence=confidence,
                    classLabel=emotion.capitalize(),
                    classId=class_id,
                    imgUID=self.image.uID
                )
                
                # Tüm skorları döndür seçeneği
                if self.return_all_scores:
                    detection.extra = {
                        "emotion_scores": face_result["emotion"],
                        "detector_used": successful_detector
                    }
                
                detection_list.append(detection)
                
        except Exception as e:
            print(f"General DeepFace error: {str(e)}")
            # Hata durumunda boş liste döndür
            return []

        return detection_list

    def run(self):
        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        
        # DeepFace kendi yüz algılamasını yapacak, detections'a ihtiyaç yok
        self.image.detections = []
        
        # DeepFace ile hem yüz algılama hem duygu analizi yap
        self.detections = self.deepface_inference()
        
        # Response'u oluştur ve döndür
        packageModel = build_response_deepface(context=self)
        return packageModel


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
