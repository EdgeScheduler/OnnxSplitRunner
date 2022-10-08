from threading import Thread
from multiprocessing import Queue,Pipe
import queue
from GPUAllocator.modelexecutor import ModelExecutor
import numpy as np
import time

def StartProcess(model_name:str,input_queue:Queue,run_signal: Pipe,done_signal: Pipe, output_queue:Queue,model_dict:dict):
    '''
    you can recv 1 by process_ok if test ok.

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

    print("init %s executor..."%model_name)

    all_threads=[]
    task_time_record=queue.Queue()
    myexecutor=ModelExecutor(model_name)
    myexecutor.Init(model_dict)

    # # run test
    # if RunTest(myexecutor,run_signal,done_signal,test_ok,input_shapes,default_batchsize):
    #     print("test run model: %s ok."%(model_name))
    # else:
    #     print("run model: %s may meet some error, fail to finish in 3 minutes, process exit."%(model_name))
    #     return

    # data-in-thread
    def read_input():
        while True:
            myexecutor.AddRequest(input_queue.get(block=True))                          # block while no input
            task_time_record.put(time.time())
    data_in_thread=Thread(target=read_input)
    all_threads.append(data_in_thread)
    data_in_thread.start()

    # data-out-thread
    def read_output():
        while True:
            result=myexecutor.modelOutput.get(block=True)                               # block while no output
            output_queue.put((model_name,result,task_time_record.get(),time.time()))
    data_out_thread=Thread(target=read_output)
    all_threads.append(data_out_thread)
    data_out_thread.start()

    # deal-data thread
    deal_thread=Thread(target=myexecutor.RunCycle, args=(run_signal,done_signal))
    all_threads.append(deal_thread)
    deal_thread.start()

    # wait all thread done. In fact, they run with no end.
    print("init %s executor ok."%model_name)
    for mythread in all_threads:
        mythread.join()

    print("process to deal %s exit."%(model_name))

# def RunTest(myexecutor: ModelExecutor, run_signal: Pipe, done_signal: Pipe,test_ok:Pipe,input_shapes:dict,default_batchsize=15)->bool:
#     success=True
#     test_input={}
#     for input_shape in input_shapes:
#         shape=[v if v>=0 else default_batchsize for v in input_shape["shape"]]
#         test_input[input_shape["name"]]=np.array(np.random.randn(*shape),dtype=input_shape["type"])

#     myexecutor.AddRequest(test_input)
#     for _ in myexecutor.modelInferenceSessions:
#         run_signal.send(1)
 
#     try:
#         myexecutor.modelOutput.get(block=True,timeout=180)
#     except Exception as ex:
#         success=False
#     finally:
#         # clear done_signal created by test
#         count=myexecutor.todo
#         if count==0:
#             count=len(myexecutor.modelInferenceSessions)
#         for _ in range(count):
#             done_signal.recv()

#     if success:
#         test_ok.send(1)
#         test_ok.close()
#         return True
#     else:
#         test_ok.send(0)
#         test_ok.close()
#         return False
