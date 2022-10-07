from Benchmark.run.timecost import evaluate_models
from SplitToChilds.moduleOperate import ModelAnalyzer
from SplitToChilds.support import SupportedModels
from config import Config
import json,os

def EvaluateModel(model_name,onnx_path=None,filename=None):
    for driver in ['CUDAExecutionProvider','CPUExecutionProvider']:
        save_fold=os.path.join("Benchmark/data/timecost/",model_name,driver.strip("ExecutionProvider"))
        os.makedirs(save_fold,exist_ok=True)
        time_evaluate_dict= evaluate_models.TimeEvaluateChildModels(model_name,driver=[driver])
        with open(os.path.join(save_fold,"data.json" if filename is None else filename),"w") as fp:
            json.dump(time_evaluate_dict,fp,indent=4)

if __name__ == "__main__":
    for model_name in SupportedModels:
        EvaluateModel(model_name,filename="current_split_5_.json")