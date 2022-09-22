import tvm

class GPUTemplate:
    target= "cuda"
    device=tvm.cuda(1)
    kind="GPU"

class CPUTemplate:
    target= "llvm"
    device=tvm.cpu(0)
    kind = "CPU"
class DeviceDriver:
    def __init__(self,kind="Unknown",target="Unknown",device=None):
        self.target= target
        self.device= device
        self.kind = kind

class CPU(DeviceDriver):
    def __init__(self,kind="",target="",device=None):
        super().__init__(CPUTemplate.kind,CPUTemplate.target,CPUTemplate.device)
        if kind !="":
            self.kind=kind

        if target !="":
            self.target=target

        if device is not None:
            self.device=device
class GPU(DeviceDriver):
    def __init__(self,kind="",target="",device=None):
        super().__init__(GPUTemplate.kind,GPUTemplate.target,GPUTemplate.device)
        if kind !="":
            self.kind=kind

        if target !="":
            self.target=target

        if device is not None:
            self.device=device
