from SplitToChilds.moduleOperate import ModelAnalyzer
from config import Config

def SplitModel(model_name,onnx_path=None):
    modelAnalyzer=ModelAnalyzer(model_name,onnx_path)
    
    # for node in modelAnalyzer:
    #     print(node.idx,node.input_info,node.output_info)

    nodes = modelAnalyzer.GetAllNodes()
    modelAnalyzer.SplitAndStoreChilds(nodes)

    # count=len(modelAnalyzer)

    # convergenceNodes = modelAnalyzer.GetConvergeNodes()
    # modelAnalyzer.SplitAndStoreChilds([nodes[count//3],nodes[count*2//3]])
    # convergenceNodes = modelAnalyzer.GetConvergeNodes()
    # modelAnalyzer.SplitAndStoreChilds(convergenceNodes)

if __name__ == "__main__":
    model_name = "googlenet"
    SplitModel(model_name)
