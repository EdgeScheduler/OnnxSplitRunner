from SplitToChilds.support import SupportedModels
from SplitToChilds import runmodule
import numpy as np
from config import Config
import time

default_batchsize=15

def Print(result:dict):
    for k,v in result.items():
        print(k,":",v[0][:10])
    print()


driver=['CUDAExecutionProvider']
test_count=10

if __name__ == "__main__":
    print("run validate:")
    model_name="yolov2"
    model_params = Config.LoadModelParamsDictById(model_name)

    print("\n==>start to validate model:",model_name)

    input_dict={}
    for value in model_params["input"]["data"]:
        shape=[v if v>=0 else default_batchsize for v in  value["shape"]]
        input_dict[value["name"]]=np.array(np.random.randn(*shape),dtype=value["type"])

    print("child")
    start=time.time()
    for _ in range(test_count):
        output=runmodule.RunChildOnnxModelSequentially(model_name,input_dict,driver)
    print((time.time()-start)/test_count)       # 3.08s 1585.875 MB  1.427s
    # Print(output)

    print("raw")
    start=time.time()
    for _ in range(test_count):
        output=runmodule.RunWholeOnnxModel(model_name,input_dict,driver)
    print((time.time()-start)/test_count)     # 1.377s 2803.875 MB
    # Print(output)