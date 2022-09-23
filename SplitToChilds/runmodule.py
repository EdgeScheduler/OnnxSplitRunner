from config import Config
import onnxruntime
import numpy as np
from typing import List,Dict
from memory_profiler import profile

def RunOnnxModelByPath(model_path: str, input_dict: dict, driver:List[str]=['CPUExecutionProvider'])->Dict[str,np.ndarray]:
    session = onnxruntime.InferenceSession(model_path,providers=driver) 

    output_labels=[v.name for v in session.get_outputs()]
    result = session.run(output_labels, input_dict)

    return {k:v for k,v in zip(output_labels,result)}

# @profile
def RunWholeOnnxModel(model_name: str, input_dict: dict, driver:List[str]=['CPUExecutionProvider'])->Dict[str,np.ndarray]:
    return RunOnnxModelByPath(Config.ModelSavePathName(model_name),input_dict,driver)


def RunChildOnnxModelByIndex(model_name: str,idx:int, input_dict: dict, driver:List[str]=['CPUExecutionProvider'])->Dict[str,np.ndarray]:
    model_path,_=Config.ChildModelSavePathName(model_name,idx)
    return RunOnnxModelByPath(model_path,input_dict,driver)

# @profile
def RunChildOnnxModelSequentially(model_name: str, input_dict: dict, driver:List[str]=['CPUExecutionProvider'])->Dict[str,np.ndarray]:
    params_dict=Config.ChildModelSumParamsDict(model_name)
    output=input_dict
    for idx in range(len(params_dict)-1): # remove key=-1 which means total-model
        output=RunChildOnnxModelByIndex(model_name,idx,output,driver)
    return output


# session = onnxruntime.InferenceSession('Onnxs/googlenet/googlenet.onnx', providers=['CUDAExecutionProvider'])
# input_data = np.array(np.random.randn(1,3,224,224)).astype(np.float32)

# session = onnxruntime.InferenceSession('Onnxs/vgg19/vgg19.onnx', providers=['CPUExecutionProvider'])
# input_data = np.array(np.random.randn(1,3,224,224)).astype(np.float32)

# print(session.get_providers())
# input_name = session.get_inputs()[0].name
# label_name = session.get_outputs()[0].name

# for _ in range(1):
#     result = session.run([label_name], {input_name: input_data})[0]
#     # print(result)

# run()
