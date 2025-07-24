
from sdks.novavision.src.helper.package import PackageHelper
from capsules.FacialEmotionRecognition.src.models.PackageModel import FacialEmotionRecognitionExecutor, PackageModel, PackageConfigs, FacialEmotionRecognitionResponse, FacialEmotionRecognitionOutputs, OutputImage, ConfigExecutor, OutputDetections

from capsules.FacialEmotionRecognition.src.models.PackageModel import FacialEmotionRecognitionDeepFaceExecutor,FacialEmotionRecognitionDeepFaceResponse, FacialEmotionRecognitionDeepFaceOutputs


def build_response(context):
    emotionOutput = OutputImage(value=context.image)
    emotionDetection = OutputDetections(value=context.prediction)
    emotionOutputs = FacialEmotionRecognitionOutputs(outputDetections=emotionDetection, outputImage=emotionOutput)
    emotionResponse = FacialEmotionRecognitionResponse(outputs=emotionOutputs)
    emotionExecutor = FacialEmotionRecognitionExecutor(value=emotionResponse)
    executor = ConfigExecutor(value=emotionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_deepface(context):
    output_image = OutputImage(value=context.image)
    output_detections = OutputDetections(value=context.image.detections)
    outputs = FacialEmotionRecognitionDeepFaceOutputs(outputImage=output_image,outputDetections=output_detections)
    response = FacialEmotionRecognitionDeepFaceResponse(outputs=outputs)
    deepface_executor = FacialEmotionRecognitionDeepFaceExecutor(value=response)
    executor = ConfigExecutor(value=deepface_executor)
    package_configs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    package_model = package.build_model(context)
    return package_model





