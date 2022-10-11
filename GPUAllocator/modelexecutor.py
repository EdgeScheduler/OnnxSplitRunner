from multiprocessing import Pipe
import onnxruntime
from typing import Tuple

class ModelExecutor:
    '''
    use deque instead of queue, because we can not read head-item without pop it from queue.
    '''
    def __init__(self,model_name: str,input_pipe: Pipe,output_pipe: Pipe, run_signal: Pipe, done_signal: Pipe):
        self.modelInferenceSessions=[]      # type：list[onnxruntime.InferenceSession]
        self.ProcessInputs=[]               # 0：Pipe, 1~n: numpy.array or None

        self.modelName=model_name
        self.modelInput=input_pipe
        self.modelOutput=output_pipe

        self.doneSignal=done_signal
        self.runSignal=run_signal

        self.ProcessOutputLabels=[]           # type: list[str]
        self.todo=0
        self.childsCount=0

    def Init(self,models_dict: dict):
        '''
        {
            "0": "xxx.onnx",
            "1": "xxx.onnx"
        }
        '''
        self.childsCount=0
        while str(self.childsCount) in models_dict:
            session=onnxruntime.InferenceSession(models_dict[str(self.childsCount)],providers=["CUDAExecutionProvider"])
            self.modelInferenceSessions.append(session)
            self.ProcessOutputLabels.append([v.name for v in session.get_outputs()])
            self.ProcessInputs.append(None)
            self.childsCount+=1
        self.ProcessInputs[0]=self.modelInput

    def Next(self):
        self.todo=(self.todo+1) % self.childsCount

    def GetTask(self)->Tuple[onnxruntime.InferenceSession, Tuple[dict,float,float], list]:
        '''
        get new task to deal. return: session, input_data, output_labels
        '''

        input_value=None
        if self.todo==0:
            input_value=self.ProcessInputs[0].recv()
        else:
            input_value=self.ProcessInputs[self.todo]
            self.ProcessInputs[self.todo]=None

        return self.modelInferenceSessions[self.todo], input_value, self.ProcessOutputLabels[self.todo]

    def EndTask(self,result: dict):
        if self.todo+1<self.childsCount:
            self.ProcessInputs[self.todo+1]=result
        else:
            self.modelOutput.send(result)

        self.Next()

    def RunCycle(self):
        print("start to run %s-cycle.\n"%self.modelName)
        while True:
            session, input_data, labels=self.GetTask()
            self.runSignal.recv()
            self.RunOnce(session, input_data, labels)
            
    def RunOnce(self,session, input_data, labels):
        '''
        Run one cycle
        '''
        if input_data is None:
            print("warning: meet no input.")
            self.doneSignal.send((self.modelName,-1))                  # send news to main-process, current task finished
            return

        result=session.run(labels,input_data)
        self.doneSignal.send((self.modelName,self.todo))               # send news to main-process, current task finished
        self.EndTask({k:v for k,v in zip(labels,result)})
