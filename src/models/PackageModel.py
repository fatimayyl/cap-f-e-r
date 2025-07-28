from typing import List, Optional, Union, Literal
from pydantic import BaseModel
from sdks.novavision.src.base.model import (
    Package, Detection as BaseDetection, Input, Output, Image, Config,
    Inputs, Configs, Outputs, Response, Request
)


# === Bounding Box ve Detection ===

class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float


class Detection(BaseDetection):
    boundingBox: BoundingBox
    imgUID: str


# === Image Wrapper ===

class ImageDetect(Image):
    detections: Optional[List[Detection]] = None


# === Inputs ===

class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Image
    type: Literal["object"] = "object"

    class Config:
        title = "Image"


# === Outputs ===

class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Image
    type: Literal["object"] = "object"

    class Config:
        title = "Image"


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


# === Configs ===

class ConfigDevice(Config):
    name: Literal["ConfigDevice"] = "ConfigDevice"
    value: Literal["CPU", "GPU"]
    type: Literal["string"] = "string"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Device"


class ConfigReturnAllScores(Config):
    name: Literal["returnAllScores"] = "returnAllScores"
    value: Literal["True", "False"]
    type: Literal["string"] = "string"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Return All Scores"


# === Input/Output Wrappers ===

class FacialEmotionRecognitionInputs(Inputs):
    inputImage: InputImage


class FacialEmotionRecognitionOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections


class FacialEmotionRecognitionConfigs(Configs):
    configDevice: ConfigDevice


class FacialEmotionRecognitionRequest(Request):
    inputs: Optional[FacialEmotionRecognitionInputs] = None
    configs: FacialEmotionRecognitionConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class FacialEmotionRecognitionResponse(Response):
    outputs: FacialEmotionRecognitionOutputs


# === DeepFace Version ===

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
        json_schema_extra = {"target": "configs"}


class FacialEmotionRecognitionDeepFaceResponse(Response):
    outputs: FacialEmotionRecognitionDeepFaceOutputs


# === Executor Configs ===

class FacialEmotionRecognitionExecutor(Config):
    name: Literal["FacialEmotionRecognition"] = "FacialEmotionRecognition"
    value: Union[FacialEmotionRecognitionRequest, FacialEmotionRecognitionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Facial Emotion Recognition"
        json_schema_extra = {"target": {"value": 0}}


class FacialEmotionRecognitionDeepFaceExecutor(Config):
    name: Literal["FacialEmotionRecognitionDeepFace"] = "FacialEmotionRecognitionDeepFace"
    value: Union[FacialEmotionRecognitionDeepFaceRequest, FacialEmotionRecognitionDeepFaceResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "DeepFace"
        json_schema_extra = {"target": {"value": 0}}


# === Package Configs ===

class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[
        FacialEmotionRecognitionExecutor,
        FacialEmotionRecognitionDeepFaceExecutor
    ]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


# === Final Package Model ===

class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["FacialEmotionRecognition"] = "FacialEmotionRecognition"
