from SplitToChilds.support import SupportedModels
from SplitToChilds.experience.testmodule import run as runmodule
import torch
from config import Config
import drivers
from load_data import easy_load_from_onnx
import time
import tvm
import importlib
import json

validate_local_onnx_file=False
mydrivers=[drivers.GPU(),drivers.CPU()]
test_count=5

if __name__ == "__main__":
    data={}
    print(">>> start to test time cost:\n")
    for model_name, config in SupportedModels.items():
        model_test_data={}
        data[model_name]=model_test_data

        print("\n==>start to test model:",model_name)
        input_shape = config["input_shape"]
        input_name = config["input_name"]

        input_data = torch.rand(*input_shape)
        input = {input_name: input_data}
        shape_dict = {input_name: input_data.shape}

        # load params
        # _, params, _ =easy_load_from_onnx(model_name,shape_dict,download_url=config["onnx_download_url"],validate_download=validate_local_onnx_file)

        for mydriver in mydrivers:
            print("=> model: {}, driver: {}:".format(model_name, mydriver))

            model_by_driver_data={}
            model_test_data[mydriver.kind]=model_by_driver_data

            model_by_driver_data["count"]=test_count
            
            # print("> onnx time:")
            # _,_,avg_time,memory_cost=runmodule.RunWholeOnnxModel(model_name,input,shape_dict,driver=mydriver,onnx_download_url=config["onnx_download_url"],validate_download=validate_local_onnx_file,count=test_count)
            # model_by_driver_data["whole_time_by_onnx"]=avg_time
            # model_by_driver_data["whole_gpu_memory_by_onnx"]=memory_cost

            print("> onnx time:")
            # _,_,avg_time,memory_cost=runmodule.RunWholeOnnxModel(model_name,input,shape_dict,driver=mydriver,onnx_download_url=config["onnx_download_url"],validate_download=validate_local_onnx_file,count=test_count)
            model_by_driver_data["whole_time_by_onnx"]=0
            model_by_driver_data["whole_gpu_memory_by_onnx"]=0

            print("> function time:")
            _,avg_time,memory_cost=runmodule.RunWholeModelByFunction(model_name,input,None,driver=mydriver,count=test_count)
            model_by_driver_data["whole_time_by_function"]=avg_time
            model_by_driver_data["whole_gpu_memory_by_function"]=memory_cost

            print("> child model time:")
            _,avg_time_list,memory_list=runmodule.RunAllChildModelSequentially(model_name,input,None,Config.ModelParamsFile(model_name=model_name),driver=mydriver,count=test_count)
            model_by_driver_data["whole_time_by_child"]={}
            model_by_driver_data["whole_gpu_memory_by_childs"]={}
            model_by_driver_data["whole_time_by_child"]["childs"]=avg_time_list
            model_by_driver_data["whole_gpu_memory_by_childs"]["childs"]=memory_list
            model_by_driver_data["whole_time_by_child"]["whole"]=sum(avg_time_list)
            model_by_driver_data["whole_gpu_memory_by_childs"]["whole"]=max(memory_list)

            with open(Config.BenchmarkDataSavePath_hot_run, "w") as fp:
                json.dump(data,fp,indent=4)