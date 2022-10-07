from cProfile import label
from collections import deque
from data import data
import onnxruntime
import numpy as np
import time
from typing import Tuple

# data append later get bigger index

class ModelExecutor:
    def __init__(self):
        self.modelInferenceSessions=[]      # type：list[onnxruntime.InferenceSession]
        self.modelInputs=[]                 # type: list[deque]
        self.modelOutputLabels=[]           # type: list[str]
        self.modelOutput=deque()
        self.inputShape=None
        self.outputShape=None
        self.todo=0

    def Init(self,models: dict):
        '''
        "0": {
            model_path: xxx.onnx
        },
        "1":{
            model_path: xxx.onnx
        }
        '''
        start=0
        while str(start) in models:
            session=onnxruntime.InferenceSession(models[str(start)]["model_path"],["CUDAExecutionProvider"])
            self.modelInferenceSessions.append(session)
            self.modelOutputLabels.append([v.name for v in session.get_outputs()])
            self.modelInputs.append(deque())

    def Next(self):
        self.todo=(self.todo+1) % len(self.modelInferenceSessions)

    def GetTask(self)->Tuple[onnxruntime.InferenceSession, dict, list]:
        '''
        get new task to deal. return: session, input_data, output_labels
        '''

        input_deque=self.modelInputs[self.todo]
        return self.modelInferenceSessions[self.todo], None if len(input_deque)<1 else input_deque[0], self.modelOutputLabels[self.todo]

    def EndTask(self,result: dict):
        input_deque=self.modelInputs[self.todo]
        output_deque=None
        if self.todo+1<len(self.modelInputs):
            output_deque=self.modelInputs[self.todo+1]
        else:
            output_deque=self.modelOutput

        input_deque.popleft()
        output_deque.append(result)

        self.Next()

    def RunCycle(self, token, signal):
        while True:
            # judge
            while not token:
                time.sleep(0.001)

                self.RunOnce()
            

    def RunOnce(self):
        session, input_data, labels=self.GetTask()
        if input_data is None:
            return
            
        result=session.run(labels,input_data)
        self.EndTask({k:v for k,v in zip(labels,result)})
