from SplitToChilds.moduleOperate import ModelAnalyzer
from config import Config

def SplitModel(model_name,onnx_path=None):
    modelAnalyzer=ModelAnalyzer(model_name,onnx_path)

    
    # for node in modelAnalyzer:
    #     print(node.idx,node.input_info,node.output_info)

    # count=len(modelAnalyzer)

    # nodes = [node for node in modelAnalyzer.GetAllNodes() if modelAnalyzer.EnableStart(node)]   

    # costs_all=[(node.idx, node.input_info["cost"]) for node in nodes]
    # costs_all=sorted(costs_all,key=lambda x:x[1],reverse=False)

    convergenceNodes = [node for node in modelAnalyzer.GetConvergeNodes() if modelAnalyzer.EnableStart(node)]
    split_nodes=[modelAnalyzer[0]]

    for node in convergenceNodes:
        if node.idx-split_nodes[-1].idx>len(modelAnalyzer)//8:
            split_nodes.append(node)


    # costs_convergence=[(node.idx, node.input_info["cost"]) for node in convergenceNodes]
    # costs_convergence=sorted(costs_convergence,key=lambda x:x[1])

    # v_all=costs_all[:len(convergenceNodes)]
    # print([v[0] for v in v_all], "sum=",sum([v[1] for v in v_all]))
    # print([v[0] for v in costs_convergence], "sum=",sum([v[1] for v in costs_convergence]))


    # split_nodes=[]
    # for node in nodes:
    #     if node.idx in [v[0] for v in v_all]:
    #         split_nodes.append(node)


    # modelAnalyzer.SplitAndStoreChilds(split_nodes)
    # convergenceNodes = modelAnalyzer.GetConvergeNodes()
    modelAnalyzer.SplitAndStoreChilds(split_nodes)

if __name__ == "__main__":
    model_name = "googlenet"
    SplitModel(model_name)
