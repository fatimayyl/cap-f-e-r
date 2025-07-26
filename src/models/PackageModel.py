from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Detection, Input, Output, Image, Config, Inputs, Configs, Outputs, Response, Request

from pydantic import BaseModel

class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float


class ImageDetect(Image):
    detections: Optional[List[Detection]] = None


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Image
    type: str = "object"

    class Config:
        title = "Image"


class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Image
    type: str = "object"

    class Config:
        title = "Image"


class FacialEmotionRecognitionInputs(Inputs):
    inputImage: InputImage


class ConfigHalfTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ConfigHalfFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


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
    """
        It refers to whether the model should run on a CPU or a GPU.
        You can select the device type for inference or training process.
    """
    name: Literal["ConfigDevice"] = "ConfigDevice"
    value: Union[ConfigDeviceCPU, ConfigDeviceGPU]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Device"


class ConfigDrawBBoxTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


#parametre
class ConfigReturnAllScoresTrue(Config):
    name: Literal["configReturnAllScoresTrue"] = "configReturnAllScoresTrue"
    value: Literal["True"] = "True"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Yes"


class ConfigReturnAllScoresFalse(Config):
    name: Literal["configReturnAllScoresFalse"] = "configReturnAllScoresFalse"
    value: Literal["False"] = "False"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "No"


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
    boundingBox: BoundingBox
    imgUID: str


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class FacialEmotionRecognitionOutputs(Outputs):
    outputImage: OutputImage
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

#buraya kadar
class FacialEmotionRecognitionDeepFaceInputs(Inputs):
    inputImage: InputImage


class FacialEmotionRecognitionDeepFaceConfigs(Configs):
    configReturnAllScores: ConfigReturnAllScores


class FacialEmotionRecognitionDeepFaceOutputs(Outputs):
    outputImage: OutputImage
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
    value: Union[FacialEmotionRecognitionExecutor,FacialEmotionRecognitionDeepFaceExecutor]
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
