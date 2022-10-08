import asyncio
import multiprocessing
import time
import numpy as np
import onnxruntime as ort
from config import Config
from threading import Thread
from multiprocessing import Process,Manager, Pool
from typing import List,Tuple

provider = "CUDAExecutionProvider"
default_batchsize=15
data_batch=30

def init_session(model_name,provider = "CUDAExecutionProvider"):
    session = ort.InferenceSession(Config.ModelSavePathName(model_name), providers=[provider])
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

def data_to_tuple(processes,batch=data_batch) ->Tuple[tuple,tuple,tuple,tuple]:
    '''
    to decrease dict read time
    '''
    model_names=tuple(list(processes.keys()))
    sessions=tuple([processes[model_name]["session"] for model_name in model_names])
    labels=tuple([processes[model_name]["output_labels"] for model_name in model_names])
    datas=tuple([tuple(processes[model_name]["datas"]) for model_name in model_names])

    return model_names,sessions,labels,datas

'''
run serially
'''
def run_serially(processes,batch=data_batch)->dict:
    model_names,sessions,labels,datas=data_to_tuple(processes,batch)

    costs=[[] for _ in model_names]

    print("=> serially, with batch=%d"%(data_batch))
    process_start=time.time()
    for batch_idx in range(batch):
        for model_id in range(len(model_names)):
            session=sessions[model_id]
            input_data=datas[model_id][batch_idx]
            label=labels[model_id]

            start=time.time()
            _ = session.run(label,input_data)
            costs[model_id].append(time.time()-start)
    proccess_cost=time.time()-process_start

    print("signal models total (s):")
    for idx in range(len(model_names)):
        print(">",model_names[idx],"(s):",sum(costs[idx]))
    print("process sum (s):",proccess_cost)
    print()

'''
run process
'''
def run_by_process_with_name(model_names,batch=data_batch, provider = "CUDAExecutionProvider"):
    def signal_process(model_name,data_batch,provider):
        model_params=Config.LoadModelParamsDictById(model_name)
        session=init_session(model_name,provider)
        input_dict={}
        for value in model_params["input"]["data"]:
            shape=[v if v>=0 else default_batchsize for v in value["shape"]]
            input_dict[value["name"]]=np.array(np.random.randn(*shape),dtype=value["type"])

        output_labels=[v["name"] for v in model_params["output"]["data"]]

        costs=[]
        for _ in range(data_batch):
            start=time.time()
            _ = session.run(output_labels,input_dict)
            costs.append(time.time()-start)
        print(">",model_name,"(s):",sum(costs))

    # enc
    print("=> by muti-process, with batch=%d"%(data_batch))
    myprocesses=[]
    process_start=time.time()
    for idx in range(len(model_names)):
        myprocess = Process(target=signal_process, args=(model_names[idx],data_batch,provider))
        myprocesses.append(myprocess)   
        myprocess.start()  
        
    for myprocess in myprocesses:
        myprocess.join()
    proccess_cost=time.time()-process_start

    print("process sum (s):",proccess_cost)
    print()
    return

# '''
# run process
# '''
# def run_by_process(processes,batch=data_batch, provider = "CUDAExecutionProvider"):
#     def signal_process(model_name,labels,datas):
#         # decode
#         print(datas)
#         inputs=[]
#         for v in datas:
#             tmp={}
#             tmp[v[0]]=np.array(v[1])
#         inputs.append(tmp)

#         inputs=[np.array(v) for v in inputs]
#         print("---",model_name,labels)
#         session=init_session(model_name,provider="CPUExecutionProvider")
#         costs=[]
#         for input_value in inputs:
#             start=time.time()
#             _ = session.run(labels,input_value)
#             costs.append(time.time()-start)
#         print(">",model_name,"(s):",sum(costs))

#     model_names,_,labels,datas=data_to_tuple(processes,batch)
#     myprocesses = []

#     # encode
#     inputs=[]
#     for model_data in datas:
#         batch_data=[]
#         for batch in model_data:
#             batch_inputs_value=[]
#             batch_inputs_key=[]
#             for key, value in batch.items():
#                 batch_inputs_key.append(key)
#                 batch_inputs_value.append(value)
#             batch_data.append([batch_inputs_key,batch_inputs_value])
#         inputs.append(batch_data)

#     print("=> by muti-process, with batch=%d"%(data_batch))
#     process_start=time.time()
#     for idx in range(len(model_names)):
#         print(inputs[idx])
#         myprocess = Process(target=signal_process, args=(model_names[idx],labels[idx], Manager().list(inputs[idx])))
#         myprocess.start()  
#         myprocesses.append(myprocess)   
#     for myprocess in myprocesses:
#         myprocess.join()
#     proccess_cost=time.time()-process_start

#     print("process sum (s):",proccess_cost)
#     print()
#     return

def run_by_thread(processes,batch=data_batch):
    def signal_thread(model_name,session,labels,inputs):
        costs=[]
        for input_value in inputs:
            start=time.time()
            _ = session.run(labels,input_value)
            costs.append(time.time()-start)
        print(">",model_name,"(s):",sum(costs))

    def other_routine(idx):
        start=time.time()
        for _ in range(3000):
            time.sleep(0.001)
        print(idx,start,time.time()-start)

    model_names,sessions,labels,datas=data_to_tuple(processes,batch)
    mythreads = []

    print("=> by threads, with batch=%d"%(data_batch))
    process_start=time.time()
    for idx in range(len(model_names)):
        thread = Thread(target=signal_thread, args=(model_names[idx],sessions[idx], labels[idx], datas[idx]))
        mythreads.append(thread)
        thread.start()
        break

    thread = Thread(target=other_routine, args=(1,))
    mythreads.append(thread)
    thread.start()

    thread = Thread(target=other_routine, args=(2,))
    mythreads.append(thread)
    thread.start()

    thread = Thread(target=other_routine, args=(3,))
    mythreads.append(thread)
    thread.start()

    for thread in mythreads:
        thread.join()

    proccess_cost=time.time()-process_start

    print("process sum (s):",proccess_cost)
    print()
    return

def run_by_routine(processes,batch=data_batch):
    async def signal_routine(model_name,session,labels,inputs):
        costs=[]
        for input_value in inputs:
            start=time.time()
            _ = session.run(labels,input_value)
            costs.append(time.time()-start)
        print(">",model_name,"(s):",sum(costs))

    async def other_routine(idx):
        start=time.time()
        for _ in range(30):
            print(idx)
            time.sleep(0.1)
        print(idx,start,time.time()-start)

    model_names,sessions,labels,datas=data_to_tuple(processes,batch)
    myroutines = []

    print("=> by routines, with batch=%d"%(data_batch))
    process_start=time.time()
    loop = asyncio.get_event_loop()
    for idx in range(len(model_names)):
        myroutines.append(signal_routine(model_names[idx],sessions[idx], labels[idx], datas[idx]))
        break

    myroutines.append(other_routine(1))
    myroutines.append(other_routine(2))
    myroutines.append(other_routine(3))

    loop.run_until_complete(asyncio.wait(myroutines))
    proccess_cost=time.time()-process_start

    print("process sum (s):",proccess_cost)
    print()
    return

def main():
    processes=load_process(["googlenet","squeezenetv1","vgg19","resnet50"])
    run_by_thread(processes)

if __name__=="__main__":
    main()