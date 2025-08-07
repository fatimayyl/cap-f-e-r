
from sdks.novavision.src.helper.package import PackageHelper
from capsules.FacialEmotionRecognition.src.models.PackageModel import FacialEmotionRecognitionExecutor, PackageModel, PackageConfigs, FacialEmotionRecognitionResponse, FacialEmotionRecognitionOutputs, OutputDetections, ConfigExecutor, OutputDetections
from capsules.FacialEmotionRecognition.src.models.PackageModel import FacialEmotionRecognitionDeepFaceExecutor, FacialEmotionRecognitionDeepFaceOutputs, FacialEmotionRecognitionDeepFaceResponse


"""
def build_response(context):
    print("DEBUG: context.prediction =", context.prediction)

    outputDetections = OutputDetections(value=context.prediction)
    facialEmotionRecognitionOutputs = FacialEmotionRecognitionOutputs(outputDetections=outputDetections)
    facialEmotionRecognitionResponse = FacialEmotionRecognitionResponse(outputs=facialEmotionRecognitionOutputs)
    facialEmotionRecognitionExecutor = FacialEmotionRecognitionExecutor(value=facialEmotionRecognitionResponse)
    executor = ConfigExecutor(value=facialEmotionRecognitionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

"""


def build_response(context):
    print("DEBUG: context.prediction =", context.prediction)
    outputDetections = OutputDetections(value=context.prediction)
    facialEmotionRecognitionOutputs = FacialEmotionRecognitionOutputs(outputDetections=outputDetections)
    facialEmotionRecognitionResponse = FacialEmotionRecognitionResponse(outputs=facialEmotionRecognitionOutputs)
    facialEmotionRecognitionExecutor = FacialEmotionRecognitionExecutor(value=facialEmotionRecognitionResponse)
    executor = ConfigExecutor(value=facialEmotionRecognitionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    print("packageModel:",packageModel)
    return packageModel



def build_response_deepface(context):
    outputDetections = OutputDetections(value=context.detections)
    facialEmotionRecognitionDeepFaceOutputs = FacialEmotionRecognitionDeepFaceOutputs(outputDetections=outputDetections ) #, outputImage=outputImage
    facialEmotionRecognitionDeepFaceResponse = FacialEmotionRecognitionDeepFaceResponse(outputs=facialEmotionRecognitionDeepFaceOutputs)
    facialEmotionRecognitionDeepFaceExecutor = FacialEmotionRecognitionDeepFaceExecutor(value=facialEmotionRecognitionDeepFaceResponse)
    executor = ConfigExecutor(value=facialEmotionRecognitionDeepFaceExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

s
