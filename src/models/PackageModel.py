from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Detection, Input, Output, Image, Config, Inputs, Configs, Outputs, Response, Request

"""
class ImageDetect(Image):
    detections: Optional[List[Detection]] = None


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Image
    type: str = "object"

    class Config:
        title = "Image"
"""


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Image
    type: str = "object"

    class Config:
        title = "Image"


class InputDetections(Input):
    name: Literal["inputDetections"] = "inputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


"""
class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Image
    type: str = "object"

    class Config:
        title = "Image"

"""


# yeni eklendi
class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection] = []
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class FacialEmotionRecognitionInputs(Inputs):
    inputImage: InputImage
    inputDetections: InputDetections


# ???????
"""
yeni actım 
"""
class ConfigHalfTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"

#???????
class ConfigHalfFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"



# facial emotion recognition için cihaz seçimi
class ConfigDeviceGPU(Config):
    name: Literal["ConfigDeviceGPU"] = "ConfigDeviceGPU"
    value: Literal["GPU"] = "GPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU"


class ConfigDeviceCPU(Config):
    name: Literal["ConfigDeviceCPU"] = "ConfigDeviceCPU"
    value: Literal["CPU"] = "CPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CPU"


class ConfigDevice(Config):
    name: Literal["ConfigDevice"] = "ConfigDevice"
    value: Union[ConfigDeviceCPU, ConfigDeviceGPU]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Device"


#yeni actım
#??????????*
class ConfigDrawBBoxTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"




class ConfigReturnAllScoresTrue(Config):
    name: Literal["configConvertToGrayTrue"] = "configConvertToGrayTrue"
    value: Literal["True"] = "True"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Yes"


class ConfigReturnAllScoresFalse(Config):
    name: Literal["configConvertToGrayFalse"] = "configConvertToGrayFalse"
    value: Literal["False"] = "False"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "No"

""" yeni kapadım
class ConfigReturnAllScoresTrue(Config):
    name: Literal["returnAllScores"] = "returnAllScores"
    value: Literal["True"] = "True"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Yes"


class ConfigReturnAllScoresFalse(Config):
    name: Literal["returnAllScores"] = "returnAllScores"
    value: Literal["False"] = "False"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "No"
"""




class ConfigReturnAllScores(Config):
    name: Literal["returnAllScores"] = "returnAllScores"
    value: Union[ConfigReturnAllScoresTrue, ConfigReturnAllScoresFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Return All Scores"


class FacialEmotionRecognitionConfigs(Configs):
    configDevice: ConfigDevice


class Detection(Detection):
    imgUID: str


class FacialEmotionRecognitionOutputs(Outputs):
    outputDetections: OutputDetections


class FacialEmotionRecognitionRequest(Request):
    inputs: Optional[FacialEmotionRecognitionInputs] = None
    configs: FacialEmotionRecognitionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class FacialEmotionRecognitionResponse(Response):
    outputs: FacialEmotionRecognitionOutputs


class FacialEmotionRecognitionExecutor(Config):
    name: Literal["FacialEmotionRecognition"] = "FacialEmotionRecognition"
    value: Union[FacialEmotionRecognitionRequest, FacialEmotionRecognitionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Facial Emotion Recognition"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


# buraya kadar
class FacialEmotionRecognitionDeepFaceInputs(Inputs):
    inputImage: InputImage
    # deepface de detection girişine gerek olmayabilir
    inputDetections: InputDetections


class FacialEmotionRecognitionDeepFaceConfigs(Configs):
    configReturnAllScores: ConfigReturnAllScores


class FacialEmotionRecognitionDeepFaceOutputs(Outputs):
    outputDetections: OutputDetections


class FacialEmotionRecognitionDeepFaceRequest(Request):
    inputs: Optional[FacialEmotionRecognitionDeepFaceInputs]
    configs: FacialEmotionRecognitionDeepFaceConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class FacialEmotionRecognitionDeepFaceResponse(Response):
    outputs: FacialEmotionRecognitionDeepFaceOutputs


class FacialEmotionRecognitionDeepFaceExecutor(Config):
    name: Literal["FacialEmotionRecognitionDeepFace"] = "FacialEmotionRecognitionDeepFace"
    value: Union[FacialEmotionRecognitionDeepFaceRequest, FacialEmotionRecognitionDeepFaceResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "DeepFace"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[FacialEmotionRecognitionExecutor, FacialEmotionRecognitionDeepFaceExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["FacialEmotionRecognition"] = "FacialEmotionRecognition"

