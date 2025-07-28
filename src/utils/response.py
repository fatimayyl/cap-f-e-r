
from sdks.novavision.src.helper.package import PackageHelper
from capsules.FacialEmotionRecognition.src.models.PackageModel import FacialEmotionRecognitionExecutor, PackageModel, PackageConfigs, FacialEmotionRecognitionResponse, FacialEmotionRecognitionOutputs, OutputImage, ConfigExecutor, OutputDetections

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


