import onnx
from config import Config

model = onnx.load(Config.ModelSavePathName("resnet50"))
# print(model)
print(model.graph.node[0])