from Onnxs.download_config import OnnxModelUrl

SupportedModels = {
    "googlenet": {
        "input_shape": (15, 3, 224, 224),
        "input_name": "data_0",
        "onnx_download_url": OnnxModelUrl.Googlenet
    },
    "resnet50": {
        "input_shape": (15, 3, 224, 224),
        "input_name": "data",
        "onnx_download_url": OnnxModelUrl.Resnet50
    },
    "squeezenetv1": {
        "input_shape": (15, 3, 224, 224),
        "input_name": "data_0",
        "onnx_download_url": OnnxModelUrl.SqueezeNetv1
    },
    "vgg19": {
        "input_shape": (15, 3, 224, 224),
        "input_name": "data",
        "onnx_download_url": OnnxModelUrl.Vgg19
    },
    "yolov2": {
        "input_shape": (15, 3, 416, 416),
        "input_name": "input.1",
        "onnx_download_url": OnnxModelUrl.Yolov2_coco
    }
}

# SupportedModels = {
#     "resnet50":{
#         "input_shape": (15, 3, 224, 224),
#         "input_name": "data",
#         "onnx_download_url": OnnxModelUrl.Resnet50
#     },
#         "googlenet": {
#         "input_shape": (15, 3, 224, 224),
#         "input_name": "data_0",
#         "onnx_download_url": OnnxModelUrl.Googlenet
#     },
# }
