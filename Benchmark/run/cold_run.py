from SplitToChilds.support import SupportedModels
from SplitToChilds.experience import runmodule
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
        _, params, _ =easy_load_from_onnx(model_name,shape_dict,download_url=config["onnx_download_url"],validate_download=validate_local_onnx_file)

        for mydriver in mydrivers:
            print("=> model: {}, driver: {}:".format(model_name, mydriver))

            model_by_driver_data={}
            model_test_data[mydriver.kind]=model_by_driver_data

            model_by_driver_data["count"]=test_count
            
            print("> onnx time")
            test_onnx_time=[]
            model_by_driver_data["whole_time_by_onnx"]=test_onnx_time
            for i in range(test_count):
                print("- run index=%d."%i)
                time_onnx=time.time()
                runmodule.RunWholeOnnxModel(model_name,input,shape_dict,driver=mydriver,onnx_download_url=config["onnx_download_url"],validate_download=validate_local_onnx_file)
                test_onnx_time.append(time.time()-time_onnx)

            print("> function time")
            test_whole_time=[]
            model_by_driver_data["whole_time_by_function"]=test_whole_time
            for i in range(test_count):
                print("- run index=%d."%i)
                time_whole=time.time()
                runmodule.RunWholeModelByFunction(model_name,input,params)
                test_whole_time.append(time.time()-time_whole)

            print("> child model time")
            test_child_time=[]
            model_by_driver_data["whole_time_by_child"]=test_child_time
            for i in range(test_count):
                print("- run index=%d."%i)
                current_time={}

                childs_time=[]
                current_time["childs"]=childs_time
                time_whole_child=time.time()
                pythonLib=importlib.import_module("ModelFuntionsPython.childs.{}".format(model_name))
                params_dict=Config.ModelParamsFile(model_name=model_name)
                output=None
                for idx in range(len(params_dict)):
                    time_run_child_model=time.time()
                    output=runmodule.RunChildModelByIdx(model_name,idx,input,params,params_dict,mydriver,output)
                    childs_time.append(time.time()-time_run_child_model)
                current_time["whole"]=time.time()-time_whole_child
                test_child_time.append(current_time)

            with open(Config.BenchmarkDataSavePath_cold_run, "w") as fp:
                json.dump(data,fp,indent=4)