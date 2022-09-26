from GPUReader import GetGPUMemoryHandle,GetGPUUsed

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