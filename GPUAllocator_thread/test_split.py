from SplitToChilds.moduleOperate import ModelAnalyzer

'''
create test model-split
'''

def SplitModel(model_name,onnx_path=None,count=1):
    modelAnalyzer=ModelAnalyzer(model_name,onnx_path)

    if model_name=="vgg19":
        modelAnalyzer.SplitAndStoreChilds([modelAnalyzer[10],modelAnalyzer[19]])
        return

    convergenceNodes = [node for node in modelAnalyzer.GetConvergeNodes() if modelAnalyzer.EnableStart(node)]
    split_nodes=[modelAnalyzer[0]]

    for node in convergenceNodes:
        if node.idx-split_nodes[-1].idx>len(modelAnalyzer)//count:
            split_nodes.append(node)

    modelAnalyzer.SplitAndStoreChilds(split_nodes)

if __name__ == "__main__":
    models={
        "vgg19": 3,
        "resnet50": 2,
        "googlenet": 1,
        "squeezenetv1": 1
    }

    for model_name, count in models.items():
        SplitModel(model_name,count=count)
