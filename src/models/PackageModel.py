from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Detection, Input, Output, Image, Config, Inputs, Configs, Outputs, Response, Request,KeyPoints


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Image
    type: str = "object"

    class Config:
        title = "Image"


class InputDetections(Input):
    name: Literal["inputDetections"] = "inputDetections"
    value: List[Detection]
    type: str = "list"

    class Config:
        title = "Detections"


class Detection(Detection):
    keyPoints: Optional[List[KeyPoints]] = None
    imgUID: str


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"



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


class ConfigReturnAllScores(Config):
    name: Literal["returnAllScores"] = "returnAllScores"
    value: Union[ConfigReturnAllScoresTrue, ConfigReturnAllScoresFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Return All Scores"





class FacialEmotionConfigs(Configs):
    configDevice: ConfigDevice

class FacialEmotionInputs(Inputs):
    inputImage: InputImage
    inputDetections: InputDetections

class FacialEmotionOutputs(Outputs):
    outputDetections: OutputDetections


class FacialEmotionRequest(Request):
    inputs: Optional[FacialEmotionInputs] = None
    configs: FacialEmotionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class FacialEmotionResponse(Response):
    outputs: FacialEmotionOutputs


class FacialEmotionExecutor(Config):
    name: Literal["FacialEmotion"] = "FacialEmotion"
    value: Union[FacialEmotionRequest, FacialEmotionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Facial Emotion"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }






class DeepFaceConfigs(Configs):
    configReturnAllScores: ConfigReturnAllScores


class DeepFaceInputs(Inputs):
    inputImage: InputImage


class DeepFaceOutputs(Outputs):
    outputDetections: OutputDetections


class DeepFaceRequest(Request):
    inputs: Optional[DeepFaceInputs]
    configs: DeepFaceConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class DeepFaceResponse(Response):
    outputs: DeepFaceOutputs


class DeepFaceExecutor(Config):
    name: Literal["DeepFace"] = "DeepFace"
    value: Union[DeepFaceRequest, DeepFaceResponse]
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
    value: Union[FacialEmotionExecutor, DeepFaceExecutor]
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

