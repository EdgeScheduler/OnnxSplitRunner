import pynvml
import time

def GetGPUMemoryHandle():
    pynvml.nvmlInit() 
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def GetGPUUsed(handle)->float:
    '''
    return MIB
    '''
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used/1048576  # MB

def Run():
    gpuHandle=GetGPUMemoryHandle()
    start_memory=GetGPUUsed(gpuHandle)

    max=-1

    while True:
        if GetGPUUsed(gpuHandle)-start_memory>max:
            max=GetGPUUsed(gpuHandle)-start_memory
            print(max,"MB")
        # time.sleep(0.001)

Run()