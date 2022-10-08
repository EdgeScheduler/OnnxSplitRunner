from multiprocessing import Pipe,Process, Queue
from threading import Thread
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
    done_signal.recv()
    vgg19.send(1)
    done_signal.recv()
    vgg19.send(2)
    done_signal.recv()
    vgg19.send(3)
    done_signal.recv()

    model_register["googlenet"]["input_queue"].put(generate_input(model_register["googlenet"]["input_shape"]))
    googlenet.send(0)
    done_signal.recv()

    model_register["resnet50"]["input_queue"].put(generate_input(model_register["resnet50"]["input_shape"]))
    resnet50.send(0)
    done_signal.recv()
    resnet50.send(1)
    done_signal.recv()
    
    model_register["squeezenetv1"]["input_queue"].put(generate_input(model_register["squeezenetv1"]["input_shape"]))
    squeezenetv1.send(0)
    done_signal.recv()
    
    for _ in model_register:
        result=model_register["output_queue"].get()
        print(result[0],"=>  cost:",result[3]-result[2],"start:",result[2],"end:",result[3])

def RunAllocator(models: list,default_batchsize=15):
    global model_register,output_queue,allocator_ready
    model_register["models"]=[]
    model_register["output_queue"]=output_queue
    done_signal_main,done_signal_child=Pipe()
    model_register["done_signal_main"]=done_signal_main
    model_register["done_signal_child"]=done_signal_child

    all_process=[]
    for model_name in models:
        model_register[model_name]={}
        model_register[model_name]["count"]=0
        model_register[model_name]["input_queue"]=Queue()
        run_signal_main,run_signal_child=Pipe()
        model_register[model_name]["run_signal_main"]=run_signal_main
        model_register[model_name]["run_signal_child"]=run_signal_child
        # process_ok_main,process_ok_child=Pipe()

        model_params=Config.ChildModelSumParamsDict(model_name)
        model_dict={}
        for idx in range(len(model_params)-1):
            model_dict[str(idx)]=model_params[str(idx)]["model_path"]
            model_register[model_name]["count"]+=1

        input_shape=[]
        for shape in model_params["-1"]["input"]["data"]:
            tmp={}
            tmp["type"]=shape["type"]
            tmp["name"]=shape["name"]
            tmp["shape"]=shape["shape"]
            input_shape.append(tmp)
        model_register[model_name]["input_shape"]=input_shape
        myprocess=Process(target=StartProcess,args=(model_name,model_register[model_name]["input_queue"],run_signal_child,done_signal_child,output_queue,model_dict))
        myprocess.start()
        all_process.append(myprocess)

        # create test
        model_register[model_name]["input_queue"].put(generate_input(input_shape,default_batchsize))
        for idx in range(model_register[model_name]["count"]):
            run_signal_main.send(idx)
            done_signal_main.recv()

        try:
            output_queue.get(block=True,timeout=180)
        except Exception as ex:
            print("run model: %s may meet some error, fail to finish in 3 minutes, process exit."%(model_name))
            if not run_signal_main.closed():
                run_signal_main.close()
            if not run_signal_child.closed():
                run_signal_child.close()
            del model_register[model_name]

            myprocess.close()
            all_process.pop()
        model_register["models"].append(model_name)
        print("test run %s ok."%model_name)
    print("success to start all executor.")
    allocator_ready=True
    for myprocess in all_process:
        myprocess.join()

def main():
    allocator_thread=Thread(target=RunAllocator,args=(["googlenet","vgg19","resnet50","squeezenetv1"],15))
    allocator_thread.start()

    data_create_thread=Thread(target=create_task)
    data_create_thread.start()

    allocator_thread.join()
    data_create_thread.join()
    

if __name__ == "__main__":
    main()