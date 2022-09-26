import pynvml
import time

gpu_hanle=None


def GetGPUMemoryHandle():
    '''
    return GPU memory Handle
    '''
    global gpu_hanle
    if gpu_hanle is not None:
        return gpu_hanle
    else:
        pynvml.nvmlInit() 
        gpu_hanle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return gpu_hanle


def GetGPUUsed(handle)->float:
    '''
    return how many GPU-Memory used (MiB)
    '''
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used/1048576  # MB