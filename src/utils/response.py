from sdks.novavision.src.helper.package import PackageHelper
from capsules.FacialEmotionRecognition.src.models.PackageModel import (
    FacialEmotionRecognitionExecutor, PackageModel, PackageConfigs,
    FacialEmotionRecognitionResponse, FacialEmotionRecognitionOutputs,
    OutputImage, ConfigExecutor, OutputDetections, Detection
)

from capsules.FacialEmotionRecognition.src.models.PackageModel import (
    FacialEmotionRecognitionDeepFaceExecutor,
    PackageModel,
    PackageConfigs,
    FacialEmotionRecognitionDeepFaceResponse,
    FacialEmotionRecognitionDeepFaceOutputs,
    OutputImage,
    OutputDetections,
    Detection,
    ConfigExecutor
)


def build_response(context):
    # Detection'lar dict olabilir, kontrol et
    detections = [
        Detection(**d) if isinstance(d, dict) else d
        for d in context.prediction
    ]

    # Output nesneleri oluştur
    emotionOutput = OutputImage(value=context.image)
    emotionDetection = OutputDetections(value=detections)
    emotionOutputs = FacialEmotionRecognitionOutputs(
        outputDetections=emotionDetection,
        outputImage=emotionOutput
    )

    # Response oluştur
    emotionResponse = FacialEmotionRecognitionResponse(outputs=emotionOutputs)

    # Executor olarak sar
    emotionExecutor = FacialEmotionRecognitionExecutor(value=emotionResponse)
    executor = ConfigExecutor(value=emotionExecutor)

    # Package ve model oluştur
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)

    return packageModel




def build_response_deepface(context):
    # context.detections: List[Detection] (zaten Detection objeleri olmalı)
    detections = []
    for d in context.detections:
        # Eğer dict ise Detection objesine çevir
        if isinstance(d, dict):
            detection = Detection(**d)
        else:
            detection = d
        detections.append(detection)

    output_image = OutputImage(name="outputImage", value=context.image)
    output_detections = OutputDetections(name="outputDetections", value=detections)

    outputs = FacialEmotionRecognitionDeepFaceOutputs(
        outputImage=output_image,
        outputDetections=output_detections
    )

    response = FacialEmotionRecognitionDeepFaceResponse(outputs=outputs)

    # Executor katmanı (DeepFace executor)
    deepface_executor = FacialEmotionRecognitionDeepFaceExecutor(value=response)
    executor = ConfigExecutor(value=deepface_executor)

    package_configs = PackageConfigs(executor=executor)
    package_helper = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)

    package_model = package_helper.build_model(context)

    return package_model
