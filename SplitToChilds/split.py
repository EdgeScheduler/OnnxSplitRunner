from SplitToChilds.moduleOperate import ModelAnalyzer
from config import Config

def SplitModel(model_name,onnx_path=None):
    modelAnalyzer=ModelAnalyzer(model_name,onnx_path)
    
    # for node in modelAnalyzer:
    #     print(node)

    nodes = modelAnalyzer.GetAllNodes()

    count=len(modelAnalyzer)

    # convergenceNodes = modelAnalyzer.GetConvergeNodes()
    modelAnalyzer.SplitAndStoreChilds([nodes[count//3],nodes[count*2//3]])

if __name__ == "__main__":
    model_name = "vgg19"
    SplitModel(model_name)
