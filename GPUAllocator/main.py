from concurrent.futures import thread
from multiprocessing import Pipe,Process, Queue
from threading import Thread
import queue
from GPUAllocator.manager import StartProcess
from config import Config
import time
import numpy as np

model_register={}
allocator_ready=False
output_queue=Queue()

def generate_input(input_shapes,default_batchsize=15)->np.array:
    new_input={}
    for input_shape in input_shapes:
        shape=[v if v>=0 else default_batchsize for v in input_shape["shape"]]
        new_input[input_shape["name"]]=np.array(np.random.randn(*shape),dtype=input_shape["type"])
    return new_input

def create_task():
    global model_register,allocator_ready
    while not allocator_ready:
        # wait all allocators ready
        time.sleep(0.01)

    # for model_name in model_register["models"]:
    #     input_shape=model_register[model_name]["input_shape"]
    #     input_queue=model_register[model_name]["input_queue"]
    #     input_queue.put(generate_input(input_shape))

    # enable run
    done_signal=model_register["done_signal_main"]
    googlenet=model_register["googlenet"]["run_signal_main"]
    resnet50=model_register["resnet50"]["run_signal_main"]
    squeezenetv1=model_register["squeezenetv1"]["run_signal_main"]
    vgg19=model_register["vgg19"]["run_signal_main"]

    model_register["vgg19"]["input_queue"].put(generate_input(model_register["vgg19"]["input_shape"]))
    vgg19.send(0)

    # googlenet come during vgg19 running. and googlenet run first
    time.sleep(0.02)
    model_register["googlenet"]["input_queue"].put(generate_input(model_register["googlenet"]["input_shape"]))
    done_signal.recv()
    googlenet.send(0)
    done_signal.recv()

    # vgg19 run again
    vgg19.send(1)
    # squeezenetv1 come during vgg19 running
    time.sleep(0.02)
    model_register["squeezenetv1"]["input_queue"].put(generate_input(model_register["squeezenetv1"]["input_shape"]))
    done_signal.recv()
    squeezenetv1.send(0)
    done_signal.recv()

    vgg19.send(2)
    # resnet50 come during vgg19 running
    time.sleep(0.007)
    model_register["resnet50"]["input_queue"].put(generate_input(model_register["resnet50"]["input_shape"]))
    done_signal.recv()
    vgg19.send(3)
    done_signal.recv()

    
    resnet50.send(0)
    # squeezenetv1 come during resnet50 running, resnet50 exit.
    time.sleep(0.008)
    model_register["squeezenetv1"]["input_queue"].put(generate_input(model_register["squeezenetv1"]["input_shape"]))
    done_signal.recv()
    squeezenetv1.send(0)
    done_signal.recv()
    resnet50.send(1)
    done_signal.recv()

    
    
    for _ in model_register:
        result=model_register["output_queue"].get()
        print(result[0],"=>  cost:",round(( result[3]-result[2])*1000)/1000.0)

def RunAllocator(models: list,input_queue: Queue,output_queue: Queue):
    model_register={}
    model_register["models"]=models
    model_register["input_queue"]=input_queue
    model_register["output_queue"]=output_queue

    done_signal_main,done_signal_child=Pipe()
    model_register["done_signal_main"]=done_signal_main
    model_register["done_signal_child"]=done_signal_child

    executor_args=[]
    communicate_dict={}
    for model_name in models:
        communicate_list=[]

        model_register[model_name]={}
        model_register[model_name]["count"]=0
        timer_queue=queue.Queue()
        model_register[model_name]["timer"]=timer_queue
        communicate_list.append(timer_queue)

        input_pipe_main, input_pipe_child=Pipe()
        model_register[model_name]["input_pipe_main"]=input_pipe_main
        model_register[model_name]["input_pipe_child"]=input_pipe_child
        communicate_list.append(input_pipe_main)

        output_pipe_main, output_pipe_child=Pipe()
        model_register[model_name]["output_pipe_main"]=output_pipe_main
        model_register[model_name]["output_pipe_child"]=output_pipe_child
        communicate_list.append(output_pipe_main)

        run_signal_main,run_signal_child=Pipe()
        model_register[model_name]["run_signal_main"]=run_signal_main
        model_register[model_name]["run_signal_child"]=run_signal_child

        model_params=Config.ChildModelSumParamsDict(model_name)
        model_dict={}
        for idx in range(len(model_params)-1):
            model_dict[str(idx)]=model_params[str(idx)]["model_path"]
            model_register[model_name]["count"]+=1

        model_register[model_name]["model_dict"]=model_dict

        input_shape=[]
        for shape in model_params["-1"]["input"]["data"]:
            tmp={}
            tmp["type"]=shape["type"]
            tmp["name"]=shape["name"]
            tmp["shape"]=shape["shape"]
            input_shape.append(tmp)
        model_register[model_name]["input_shape"]=input_shape

        communicate_dict[model_name]=communicate_list
        executor_args.append([model_name,input_pipe_child,output_pipe_child,run_signal_child,done_signal_child,model_dict])

    def deal_process(args: list):
        '''
        run all executor in threads with one independent Process.
        '''
        threads=[]
        for arg in args:
            mythread=Thread(target=StartProcess,args=(arg))
            mythread.start()
            threads.append(mythread)

        for mythread in threads:
            mythread.join()

    run_deal_process=Process(target=deal_process,args=(executor_args,))
    run_deal_process.start()

    def assign_task(input_queue: Queue,output_queue: Queue, communicate_dict:dict):
        '''
        model_dict: 
        {
            "$model_name": [timer, input_pipe_main, output_pipe_main]
        }
        '''
        def give_input(input_queue,communicate_dict):
            while True:
                task=input_queue.get()
                model_name,data=task[0],task[1]
                input_pipe=communicate_dict[model_name][1]
                timer=communicate_dict[model_name][0]
                timer.put(time.time())
                input_pipe.send(data)
        def read_output(model_name,output_pipe):
            while True:
                result=output_pipe.recv()
                end_time=time.time()
                start_time=communicate_dict[model_name][0].get()
                output_queue.put((model_name,result,start_time,end_time))
        
        threads=[]
        input_thread=Thread(target=give_input,args=(input_queue,communicate_dict))
        input_thread.start()

        threads.append(input_thread)

        for model_name in communicate_dict:
            mythread=Thread(target=read_output,args=(model_name,communicate_dict[model_name][2]))
            mythread.start()
            threads.append(mythread)
            
        for mythread in threads:
            mythread.join()

    run_assign_task=Process(target=assign_task,args=(input_queue,output_queue,communicate_dict))
    run_assign_task.start()

    return model_register,run_deal_process,run_assign_task

def RunTest(model_register,input_queue,output_queue):
    print("-------test system-------")
    for model_name in model_register["models"]:
        count = model_register[model_name]["count"]
        input_shape = model_register[model_name]["input_shape"]
        run_signal= model_register[model_name]["run_signal_main"]
        done_signal= model_register["done_signal_main"]

        input_queue.put((model_name,generate_input(input_shape)))
        for _ in range(count):
            run_signal.send(1)
            done_signal.recv()

        result=output_queue.get()
        print("test %s ok, finished in %fs"%(result[0],round(result[3]*10000-result[2]*10000)/10000))
    print("----------end----------")

def main():
    input_queue=Queue()
    output_queue=Queue()
    model_register,run_deal_process,run_assign_task=RunAllocator(["googlenet","vgg19","resnet50","squeezenetv1"],input_queue,output_queue)

    for _ in range(10):
        RunTest(model_register,input_queue,output_queue)

    run_deal_process.join()
    run_assign_task.join()

    # data_create_thread=Thread(target=create_task)
    # data_create_thread.start()
    # data_create_thread.join()
    

if __name__ == "__main__":
    main()