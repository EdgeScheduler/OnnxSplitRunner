from config import Config
import onnxruntime
from SplitToChilds.support import SupportedModels
from SplitToChilds import runmodule
import numpy as np
import time
from typing import List,Dict

def TimeEvaluateChildModels(model_name:str,driver:List[str]=['CPUExecutionProvider'], test_count=20,default_batchsize=15)->Dict[str,Dict[str,float]]:
    if model_name not in SupportedModels:
        print("unknown model, its not registered.")
        return None

    result={
        "cold-run-mode":{
            "raw": {}, 
            "sequentially": {},
            "childs": {}
        },
        "throughout-mode": {
            "raw": {}, 
            # "sequentially": {},
            "childs": {}
        }
    }
    print("start to evaluate model-%s, this may take some time."%(model_name))
    model_params = Config.LoadModelParamsDictById(model_name)
    params_dict=Config.ChildModelSumParamsDict(model_name)

    input_dict={}
    for value in model_params["input"]["data"]:
        shape=[v if v>=0 else default_batchsize for v in value["shape"]]
        input_dict[value["name"]]=np.array(np.random.randn(*shape),dtype=value["type"])

    ####################################################################################################

    ##### evaluate cold-run-mode #####
    print("start to evaluate cold-run-mode...")

    '''
    raw
    '''
    # first-time
    start=time.time()
    runmodule.RunWholeOnnxModel(model_name,input_dict,driver)  
    result["cold-run-mode"]["raw"]["first"]=time.time()-start

    # avg
    start=time.time()
    for _ in range(test_count):
        runmodule.RunWholeOnnxModel(model_name,input_dict,driver)  
    result["cold-run-mode"]["raw"]["avg"]=(time.time()-start)/test_count
    result["cold-run-mode"]["raw"]["from"]=params_dict["-1"]["from"]
    result["cold-run-mode"]["raw"]["to"]=params_dict["-1"]["to"]

    '''
    sequentially
    '''
    # first-time
    start=time.time()
    runmodule.RunChildOnnxModelSequentially(model_name,input_dict,driver)  
    result["cold-run-mode"]["sequentially"]["first"]=time.time()-start

    # avg
    start=time.time()
    for _ in range(test_count):
        runmodule.RunChildOnnxModelSequentially(model_name,input_dict,driver)  
    result["cold-run-mode"]["sequentially"]["avg"]=(time.time()-start)/test_count
    result["cold-run-mode"]["sequentially"]["from"]=params_dict["-1"]["from"]
    result["cold-run-mode"]["sequentially"]["to"]=params_dict["-1"]["to"]

    '''
    childs
    '''
    output=input_dict
    for idx in range(len(params_dict)-1): # remove key=-1 which means total-model
        result["cold-run-mode"]["childs"][str(idx)]={}
        start=time.time()
        tmp=runmodule.RunChildOnnxModelByIndex(model_name,idx,output,driver)
        result["cold-run-mode"]["childs"][str(idx)]["first"]=time.time()-start

        start=time.time()
        for _ in range(test_count):
            tmp=runmodule.RunChildOnnxModelByIndex(model_name,idx,output,driver) 
        result["cold-run-mode"]["childs"][str(idx)]["avg"]=(time.time()-start)/test_count
        result["cold-run-mode"]["childs"][str(idx)]["from"]=params_dict[str(idx)]["from"]
        result["cold-run-mode"]["childs"][str(idx)]["to"]=params_dict[str(idx)]["to"]
        output=tmp

    ####################################################################################################



    ####################################################################################################

    ##### evaluate throughout-mode #####
    test_count*=3       # increase test_count
    print("start to evaluate throughout-mode...")

    '''
    raw
    '''
    session = onnxruntime.InferenceSession(Config.ModelSavePathName(model_name),providers=driver) 

    output_labels=[v.name for v in session.get_outputs()]

    session.run(output_labels, input_dict)
    # avg
    start=time.time()
    for _ in range(test_count):
        session.run(output_labels, input_dict) 
    result["throughout-mode"]["raw"]["avg"]=(time.time()-start)/test_count
    result["throughout-mode"]["raw"]["from"]=params_dict["-1"]["from"]
    result["throughout-mode"]["raw"]["to"]=params_dict["-1"]["to"]

    # '''
    # sequentially
    # '''
    # # avg
    # start=time.time()
    # for _ in range(test_count):
    #     runmodule.RunChildOnnxModelSequentially(model_name,input_dict,driver)  
    # result["throughout-mode"]["sequentially"]["avg"]=(time.time()-start)/test_count

    '''
    childs
    '''
    
    output=input_dict
    for idx in range(len(params_dict)-1): # remove key=-1 which means total-model
        result["throughout-mode"]["childs"][str(idx)]={}

        model_path,_=Config.ChildModelSavePathName(model_name,idx)
        session = onnxruntime.InferenceSession(model_path,providers=driver) 
        output_labels=[v.name for v in session.get_outputs()]
        tmp=session.run(output_labels, output)
        
        start=time.time()
        for _ in range(test_count):
            tmp=session.run(output_labels, output) 
        result["throughout-mode"]["childs"][str(idx)]["avg"]=(time.time()-start)/test_count
        result["throughout-mode"]["childs"][str(idx)]["from"]=params_dict[str(idx)]["from"]
        result["throughout-mode"]["childs"][str(idx)]["to"]=params_dict[str(idx)]["to"]
        
        output={k:v for k,v in zip(output_labels,tmp)}

    ####################################################################################################

    print("success to evaluate model-%s."%(model_name))
    return result