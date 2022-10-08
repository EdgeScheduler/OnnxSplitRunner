from collections import deque
from queue import Queue
import onnxruntime
import time
from typing import Tuple

# data append later get bigger index

class ModelExecutor:
    '''
    use deque instead of queue, because we can not read head-item without pop it from queue.
    '''
    def __init__(self,model_name):
        self.modelInferenceSessions=[]      # type：list[onnxruntime.InferenceSession]
        self.modelInputs=[]                 # type: list[deque]
        self.modelOutputLabels=[]           # type: list[str]
        self.modelInput=None                # 
        self.modelOutput=Queue()            # type: Queue
        self.todo=0
        self.model_name=model_name

    def Init(self,models: dict):
        '''
        {
            "0": "xxx.onnx",
            "1": "xxx.onnx"
        }
        '''
        start=0
        while str(start) in models:
            session=onnxruntime.InferenceSession(models[str(start)],providers=["CUDAExecutionProvider"])
            self.modelInferenceSessions.append(session)
            self.modelOutputLabels.append([v.name for v in session.get_outputs()])
            self.modelInputs.append(deque())
            start+=1
        self.modelInput=self.modelInputs[0]

    def AddRequest(self,input_dict):
        self.modelInput.append(input_dict)

    def Next(self):
        self.todo=(self.todo+1) % len(self.modelInferenceSessions)

    def GetTask(self)->Tuple[onnxruntime.InferenceSession, Tuple[dict,float,float], list]:
        '''
        get new task to deal. return: session, input_data, output_labels
        '''
        input_deque=self.modelInputs[self.todo]
        input_value=None
        start=time.time()
        while len(input_deque) < 1 and time.time()-start<3:
            pass

        if len(input_deque)>0:
            input_value=input_deque[0]

        return self.modelInferenceSessions[self.todo], input_value, self.modelOutputLabels[self.todo]

    def EndTask(self,result: dict):
        self.modelInputs[self.todo].popleft()
        if self.todo+1<len(self.modelInputs):
            self.modelInputs[self.todo+1].append(result)
        else:
            self.modelOutput.put(result)

        self.Next()

    def RunCycle(self, run_signal,done_signal):
        while True:
            run_signal.recv()
            self.RunOnce(done_signal)
            
    def RunOnce(self,done_signal):
        '''
        Run one cycle
        '''
        session, input_data, labels=self.GetTask()
        if input_data is None:
            print("warning: meet no input.")
            done_signal.send((self.model_name,-1))                  # send news to main-process, current task finished
            return

        result=session.run(labels,input_data)
        done_signal.send((self.model_name,self.todo))               # send news to main-process, current task finished
        self.EndTask({k:v for k,v in zip(labels,result)})

    def IsFree(self)->bool:
        if len(self.modelInputs)<1 and self.todo==0:
            return True
        else:
            return False
