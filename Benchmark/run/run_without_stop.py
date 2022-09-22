from SplitToChilds.experience import runmodule,export2lib
from SplitToChilds.support import SupportedModels
import drivers
from config import Config
import torch
import pynvml
import tvm
from tvm import relay
from tvm.contrib import graph_executor

def Add():
    x = relay.var("input_x", shape=(1,), dtype="float32")
    y = relay.var("input_y", shape=(1,), dtype="float32")
    f = relay.add(x, y)

    return f

def GetGPUMemoryHandle():
    pynvml.nvmlInit() 
    return pynvml.nvmlDeviceGetHandleByIndex(1)

def GetGPUUsed(handle)->float:
    '''
    return MIB
    '''
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used/1048576  # MB

mydriver=drivers.GPU()

# test TVM-runtime
gpuHandle=GetGPUMemoryHandle()
raw_memory=GetGPUUsed(gpuHandle)

start_memory=raw_memory

ir_module = tvm.IRModule.from_expr(Add())
with tvm.transform.PassContext(opt_level=5):
    lib = relay.build(ir_module, mydriver.target)
module = graph_executor.GraphModule(lib["default"](mydriver.device))
module.set_input("input_x",torch.rand(1))
module.set_input("input_y",torch.rand(1))
module.run()
new_memory=GetGPUUsed(gpuHandle)
print("TVM Runtime cost:",GetGPUUsed(gpuHandle)-raw_memory,"MB", "; start={}MB, end={}MB".format(raw_memory,new_memory))
start_memory=new_memory

model_names=['googlenet']

for model_name in model_names:

    input_shape = SupportedModels[model_name]["input_shape"]
    input_name = SupportedModels[model_name]["input_name"]

    max_value=0
    # for _ in range(10000):
    #     idx=5
    #     input_dict=export2lib.LoadInputDict(model_name,mydriver,idx)
    #     input_name=input_dict.keys()[0]     # type: str
    #     input_data = torch.rand(*input_dict[input_name])
    #     gpuHandle=GetGPUMemoryHandle()
    #     # runmodule.RunAllChildModelSequentially(model_name,{input_name: input_data},None,Config.ModelParamsFile(model_name=model_name),driver=mydriver)
    #     # print("1->:",GetGPUUsed(gpuHandle)-start_memory,"MB")

    #     runmodule.RunChildModelByIdx(model_name,0,{input_name: input_data},None,Config.ModelParamsFile(model_name=model_name),driver=mydriver)
    #     runmodule.RunWholeModelByFunction(model_name,{input_name: input_data},None,driver=mydriver)
    #     print("2->:",GetGPUUsed(gpuHandle)-start_memory,"MB")

    #     # if GetGPUUsed(gpuHandle)-start_memory>max_value:
    #     #     max_value=GetGPUUsed(gpuHandle)-start_memory
    #     # print(max_value,"MB")

    for _ in range(10):
        idx=13
        input_dict=export2lib.LoadInputDict(model_name,mydriver,idx)

        for k in input_dict:
            input_dict[k]=torch.rand(*input_dict[k])

        gpuHandle=GetGPUMemoryHandle()

        runmodule.RunChildModelByIdx(model_name,idx,input_dict,None,Config.ModelParamsFile(model_name=model_name),driver=mydriver)

        print("2->:",GetGPUUsed(gpuHandle)-start_memory,"MB", "; total: ",GetGPUUsed(gpuHandle)-raw_memory,"MB")
