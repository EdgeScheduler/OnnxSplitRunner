import onnxruntime
import numpy as np
from memory_profiler import profile
# onnxruntime=1.12.1


### 推理

@profile
def run():
    # CUDAExecutionProvider, CPUExecutionProvider

    
    session0 = onnxruntime.InferenceSession('Onnxs/vgg19/childs/0/vgg19-0.onnx', providers=['CUDAExecutionProvider']) 
    session1 = onnxruntime.InferenceSession('Onnxs/vgg19/childs/1/vgg19-1.onnx', providers=['CUDAExecutionProvider']) 
    session2 = onnxruntime.InferenceSession('Onnxs/vgg19/childs/2/vgg19-2.onnx', providers=['CUDAExecutionProvider']) 

    session = onnxruntime.InferenceSession('Onnxs/vgg19/vgg19.onnx', providers=['CUDAExecutionProvider']) 

    if session0 is not None and session1 is not None and session2 is not None and session is not None:
        print("end")

run()