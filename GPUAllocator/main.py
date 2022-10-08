from multiprocessing import Pipe,Process, Queue
from threading import Thread, thread
from GPUAllocator.manager import StartProcess
from config import Config
import time

model_register={}
allocator_ready=False
output_queue=Queue()

def create_task():
    global model_register,allocator_ready
    while not allocator_ready:
        # wait all allocators ready
        time.sleep(0.01)

    

def RunAllocator(models: list,default_batchsize=15):
    global model_register,output_queue,allocator_ready
    model_name["output_queue"]=output_queue
    done_signal_main,done_signal_child=Pipe()
    model_name["done_signal_main"]=done_signal_main
    model_name["done_signal_child"]=done_signal_child

    all_process=[]
    for model_name in models:
        model_register[model_name]={}
        model_register[model_name]["input_queue"]=Queue()
        run_signal_main,run_signal_child=Pipe()
        model_register[model_name]["run_signal_main"]=run_signal_main
        model_register[model_name]["run_signal_child"]=run_signal_child
        process_ok_main,process_ok_child=Pipe()

        model_params=Config.ChildModelSumParamsDict()
        model_dict={}
        for idx in range(len(model_params)-1):
            model_dict[str(idx)]=model_params[str(idx)]["model_path"]

        input_shape=[]
        for shape in model_params["-1"]["input"]["data"]:
            tmp={}
            tmp["type"]=shape["type"]
            tmp["name"]=shape["name"]
            tmp["shape"]=shape["shape"]
            input_shape.append(tmp)

        myprocess=Process(target=StartProcess,args=(model_name,model_register[model_name]["input_queue"],run_signal_child,done_signal_child,process_ok_child,output_queue,model_dict,input_shape,default_batchsize))
        myprocess.start()
        all_process.append(myprocess)

        if process_ok_main.recv()==0:      # block util model test ok.
            if not run_signal_main.closed():
                run_signal_main.close()
            if not run_signal_child.closed():
                run_signal_child.close()
            del model_register[model_name]
        if not process_ok_main.closed():
            process_ok_main.close()

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