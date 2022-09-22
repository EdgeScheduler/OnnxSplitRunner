from SplitToChilds.support import SupportedModels
from SplitToChilds import split

if __name__ == "__main__":
    print("run split:")
    for model_name in SupportedModels.keys():
        print("\n==>start to split model:",model_name)
        split.SplitModel(model_name)
        