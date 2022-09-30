import onnxruntime as ort
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../"))
import asyncio
import multiprocessing
import time
import numpy as np
import onnxruntime as ort
from config import Config
from threading import Thread
from multiprocessing import Process,Manager, Pool
from typing import List,Tuple
import os

provider = "CUDAExecutionProvider"
default_batchsize=15
data_batch=300

def init_session(model_name,provider = "CUDAExecutionProvider"):
    session = ort.InferenceSession(os.path.join(os.path.dirname(os.path.abspath(__file__)),model_name+".onnx"), providers=[provider])
    return session

def load_process(model_names:List[str], provider = "CUDAExecutionProvider")->dict:
    # 为了减少读视频的时间，复制相同的图片组成batch
    processes={}
    for model_name in model_names:
        session=init_session(model_name,provider)

        input_datas=[]

        model_params=Config.LoadModelParamsDictById(model_name)

        for _ in range(data_batch):
            input_dict={}
            for value in model_params["input"]["data"]:
                shape=[v if v>=0 else default_batchsize for v in value["shape"]]
                input_dict[value["name"]]=np.array(np.random.randn(*shape),dtype=value["type"])
            input_datas.append(input_dict)

        processes[model_name]={}
        processes[model_name]["session"]=session
        processes[model_name]["datas"]=tuple(input_datas)
        processes[model_name]["output_labels"]=[v["name"] for v in model_params["output"]["data"]]

    return processes

def signal_process(model_name,data_batch=data_batch,provider = "CUDAExecutionProvider"):
    print("=>%s by muti-process, with batch=%d"%(model_name,data_batch))
    model_params=Config.LoadModelParamsDictById(model_name)
    session=init_session(model_name,provider)
    input_dict={}
    for value in model_params["input"]["data"]:
        shape=[v if v>=0 else default_batchsize for v in value["shape"]]
        input_dict[value["name"]]=np.array(np.random.randn(*shape),dtype=value["type"])

    output_labels=[v["name"] for v in model_params["output"]["data"]]

    costs=[]
    process_start=time.time()
    for _ in range(data_batch):
        start=time.time()
        _ = session.run(output_labels,input_dict)
        costs.append(time.time()-start)
    proccess_cost=time.time()-process_start

    print(">",model_name,"(s):",sum(costs))
    print(model_name,"process sum (s):",proccess_cost)


def main():
    signal_process("squeezenetv1")

if __name__=="__main__":
    main()