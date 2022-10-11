from threading import Thread
from multiprocessing import Queue,Pipe
import queue
from GPUAllocator.modelexecutor import ModelExecutor
import numpy as np
import time

def StartProcess(model_name: str,input_pipe: Pipe,output_pipe: Pipe, run_signal_pipe: Pipe, done_signal_pipe: Pipe,model_dict:dict):
    '''
    model-dict example:
        {
            "0": "xxx.onnx",
            "1": "xxx.onnx"
        }

    input_shapes example:

        [
            {
                "type": "float32",
                "name": ...,
                "shape": ...
            },
            {
                ...
            }
        ]

    '''

    myexecutor=ModelExecutor(model_name,input_pipe,output_pipe, run_signal_pipe, done_signal_pipe)
    myexecutor.Init(model_dict)
    myexecutor.RunCycle()