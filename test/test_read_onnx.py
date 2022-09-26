import onnx
import onnx.utils
from config import Config

model = onnx.load(Config.ModelSavePathName("googlenet"))
# print(model)
# print(model.graph.input)
# print(model.graph.output)
for node in model.graph.value_info:
    print(node)
    break
# onnx.utils.extract_model(Config.ModelSavePathName("googlenet"),"tmp.onnx", ["data_0"], ["onnx::Concat_480"])