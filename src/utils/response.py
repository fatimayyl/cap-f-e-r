from sdks.novavision.src.helper.package import PackageHelper
from capsules.FacialEmotionRecognition.src.models.PackageModel import FacialEmotionExecutor, PackageModel, PackageConfigs, FacialEmotionResponse, FacialEmotionOutputs, OutputDetections, ConfigExecutor, OutputDetections
from capsules.FacialEmotionRecognition.src.models.PackageModel import DeepFaceExecutor, DeepFaceOutputs, DeepFaceResponse




def build_response(context):
    outputDetections = OutputDetections(value=context.prediction)
    facialEmotionOutputs = FacialEmotionOutputs(outputDetections=outputDetections)
    facialEmotionResponse = FacialEmotionResponse(outputs=facialEmotionOutputs)
    facialEmotionExecutor = FacialEmotionExecutor(value=facialEmotionResponse)
    executor = ConfigExecutor(value=facialEmotionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel



def build_response_deepface(context):
    outputDetections = OutputDetections(value=context.detections)
    deepFaceOutputs = DeepFaceOutputs(outputDetections=outputDetections ) #, outputImage=outputImage
    deepFaceResponse = DeepFaceResponse(outputs=deepFaceOutputs)
    deepFaceExecutor = DeepFaceExecutor(value=deepFaceResponse)
    executor = ConfigExecutor(value=deepFaceExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel
