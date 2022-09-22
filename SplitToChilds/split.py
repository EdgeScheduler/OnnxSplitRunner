from SplitOnnxToChilds.moduleOperate import ModelAnalyzer
from config import Config

def SplitModel(model_name,onnx_path=None):
    modelAnalyzer=ModelAnalyzer(model_name,onnx_path)
    
    # for node in modelAnalyzer:
    #     print(node)

    # convergenceNodes = modelAnalyzer.GetAllNodes()
    convergenceNodes = modelAnalyzer.GetConvergeNodes()
    modelAnalyzer.SplitAndStoreChilds(convergenceNodes)

if __name__ == "__main__":
    model_name = "resnet50"
    SplitModel(model_name)
