<<<<<<< HEAD
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Detection, Input, Output, Image, Config, Inputs, Configs, Outputs, Response, Request,KeyPoints
=======

from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config
>>>>>>> 7f2e0768e16a9e2031f66856f03d3372b86a18b4


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
<<<<<<< HEAD
    value: Image
    type: str = "object"

=======
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

>>>>>>> 7f2e0768e16a9e2031f66856f03d3372b86a18b4
    class Config:
        title = "Image"


<<<<<<< HEAD
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
=======
class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image],Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class KeepSideFalse(Config):
>>>>>>> 7f2e0768e16a9e2031f66856f03d3372b86a18b4
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


<<<<<<< HEAD

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
=======
class KeepSideTrue(Config):
>>>>>>> 7f2e0768e16a9e2031f66856f03d3372b86a18b4
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


<<<<<<< HEAD


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
=======
class KeepSideBBox(Config):
    """
        Rotate image without catting off sides.
    """
    name: Literal["KeepSide"] = "KeepSide"
    value: Union[KeepSideTrue, KeepSideFalse]
>>>>>>> 7f2e0768e16a9e2031f66856f03d3372b86a18b4
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
<<<<<<< HEAD
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
=======
        title = "Keep Sides"


class Degree(Config):
    """
        Positive angles specify counterclockwise rotation while negative angles indicate clockwise rotation.
    """
    name: Literal["Degree"] = "Degree"
    value: int = Field(ge=-359.0, le=359.0,default=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[-359, 359]"] = "[-359, 359]"

    class Config:
        title = "Angle"


class PackageInputs(Inputs):
    inputImage: InputImage


class PackageConfigs(Configs):
    degree: Degree
    drawBBox: KeepSideBBox


class PackageOutputs(Outputs):
    outputImage: OutputImage


class PackageRequest(Request):
    inputs: Optional[PackageInputs]
    configs: PackageConfigs
>>>>>>> 7f2e0768e16a9e2031f66856f03d3372b86a18b4

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


<<<<<<< HEAD
class FacialEmotionResponse(Response):
    outputs: FacialEmotionOutputs


class FacialEmotionExecutor(Config):
    name: Literal["FacialEmotion"] = "FacialEmotion"
    value: Union[FacialEmotionRequest, FacialEmotionResponse]
=======
class PackageResponse(Response):
    outputs: PackageOutputs


class PackageExecutor(Config):
    name: Literal["Package"] = "Package"
    value: Union[PackageRequest, PackageResponse]
>>>>>>> 7f2e0768e16a9e2031f66856f03d3372b86a18b4
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
<<<<<<< HEAD
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
=======
        title = "Package"
>>>>>>> 7f2e0768e16a9e2031f66856f03d3372b86a18b4
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
<<<<<<< HEAD
    value: Union[FacialEmotionExecutor, DeepFaceExecutor]
=======
    value: Union[PackageExecutor]
>>>>>>> 7f2e0768e16a9e2031f66856f03d3372b86a18b4
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
<<<<<<< HEAD
=======
        json_schema_extra = {
            "target": "value"
        }
>>>>>>> 7f2e0768e16a9e2031f66856f03d3372b86a18b4


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
<<<<<<< HEAD
    type: Literal["capsule"] = "capsule"
    name: Literal["FacialEmotionRecognition"] = "FacialEmotionRecognition"

=======
    type: Literal["component"] = "component"
    name: Literal["Package"] = "Package"
>>>>>>> 7f2e0768e16a9e2031f66856f03d3372b86a18b4
