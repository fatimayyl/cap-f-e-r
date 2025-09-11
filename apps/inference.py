
import os
import sys
import cv2
import json
import requests
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))



from sdks.novavision.src.media.image import Image as image
from sdks.novavision.src.base.model import BoundingBox, Detection
from capsules.FacialEmotionRecognition.src.models.PackageModel import PackageModel, PackageConfigs, FacialEmotionRecognitionConfigs, FacialEmotionRecognitionInputs, FacialEmotionRecognitionExecutor, FacialEmotionRecognitionRequest, ImageDetect, InputImage, ConfigExecutor, ConfigDevice, ConfigDeviceCPU, ConfigDeviceGPU

ENDPOINT_URL = "http://127.0.0.1:8000/api"


def inference():
    boundingBox1 = BoundingBox(left=200, top=50, width=250, height=250)
    detection1 = Detection(boundingBox=boundingBox1, confidence=0.5, classLabel="Face", classId=0, imgUID="323332")
    image_face = ImageDetect(name="image", detections=[detection1], uID="323332", mimeType="image/jpg",
                             encoding="base64", value=np.asarray(cv2.imread("/opt/project/capsules/FacialEmotionRecognition/resources/img4.jpg")).astype(np.float32),
                             type="Image")
    image_face = image.encode64(image_face)
    configDeviceGPU = ConfigDeviceGPU(configHalf='GPU')
    configDeviceCPU = ConfigDeviceCPU(value="CPU")
    configDevice = ConfigDevice(value=configDeviceCPU)
    facialEmotionRecognitionConfigs = FacialEmotionRecognitionConfigs(configDevice=configDevice)
    inputImage = InputImage(value=image_face)
    facialEmotionRecognitionInputs = FacialEmotionRecognitionInputs(inputImage=inputImage)
    facialEmotionRecognitionRequest = FacialEmotionRecognitionRequest(inputs=facialEmotionRecognitionInputs, configs=facialEmotionRecognitionConfigs)
    facialEmotionRecognitionExecutor = FacialEmotionRecognitionExecutor(value=facialEmotionRecognitionRequest)
    executor = ConfigExecutor(value=facialEmotionRecognitionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    request = PackageModel(configs=packageConfigs, name="FacialEmotionRecognition")
    request_json = json.loads(request.json())
    response = requests.post(ENDPOINT_URL, json=request_json)
    return response.json()


if __name__ == "__main__":
    inference()
