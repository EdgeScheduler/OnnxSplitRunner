from Benchmark.run.timecost import evaluate_models
from SplitToChilds.moduleOperate import ModelAnalyzer
from SplitToChilds.support import SupportedModels
from config import Config
import json,os

'''
please run "Benchmark/run/timecost/evaluate_minimum_split.py" before to get bench, then you can run "Benchmark/run/timecost/evaluate_current_split.py" to evaluate current split
'''

def EvaluateModel(model_name,count=4,onnx_path=None):
    for driver in ['CPUExecutionProvider']:
        with open(os.path.join(os.path.join("Benchmark/data/timecost/",model_name,driver.strip("ExecutionProvider")),"data.json"),"r") as fp:
            data=json.load(fp)

        print("\n\nraw=>",model_name,":",data["throughout-mode"]["raw"]["avg"],"s")
        child_data=data["throughout-mode"]["childs"]

        result=[]

        time_datas=[]
        for idx in range(len(child_data)):
            time_datas.append(child_data[str(idx)]["avg"])

        total=sum(time_datas)
        print("sum=>",model_name,":",total,"s")

        child_total=total/count

        tmp=0.0
        for idx,current_time in enumerate(time_datas):
            tmp+=current_time
            if tmp>=child_total:
                print(tmp)
                result.append(idx+1)
                tmp=0
        print(tmp)

        if result[0]!=0:
            result=[0]+result
        print("split idx:",[child_data[str(idx)]["from"] for idx in result],"\n")

        modelAnalyzer=ModelAnalyzer(model_name,onnx_path)
        modelAnalyzer.SplitAndStoreChilds([modelAnalyzer[i] for i in [child_data[str(idx)]["from"] for idx in result]])


if __name__ == "__main__":
    for model_name in SupportedModels:
        EvaluateModel(model_name,count=4)
