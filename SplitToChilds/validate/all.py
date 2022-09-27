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
    for model_name in SupportedModels:
        model_params = Config.LoadModelParamsDictById(model_name)

        print("\n==>start to validate model:",model_name)

        input_dict={}
        for value in model_params["input"]["data"]:
            shape=[v if v>=0 else default_batchsize for v in value["shape"]]
            input_dict[value["name"]]=np.array(np.random.randn(*shape),dtype=value["type"])

        output=runmodule.RunWholeOnnxModel(model_name,input_dict,driver)                
        print("raw",str(output)[:100])
        output=runmodule.RunChildOnnxModelSequentially(model_name,input_dict,driver)  
        print("child",str(output)[:100])

        print()
        start=time.time()
        for _ in range(test_count):
            output=runmodule.RunWholeOnnxModel(model_name,input_dict,driver)
        print("raw time:",(time.time()-start)/test_count) 

        start=time.time()
        for _ in range(test_count):
            output=runmodule.RunChildOnnxModelSequentially(model_name,input_dict,driver)
        print("child time:",(time.time()-start)/test_count) 

'''
googlenet:      0.19371s/0.19364s               1151MB/739MB  
resnet50:       0.2826ss/0.378s                 1105MB/865MB
squeezenetv1:   0.084450s/0.155280s             879MB/865MB
vgg19:          1.237s/1.515                    2705MB/1649MB
YOLOv2:         0.4175s/0.560391s               1108/885MB
'''