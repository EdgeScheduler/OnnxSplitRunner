import os
from typing import Dict,List,Tuple
import json

class Config:
    # static path property
    ProjectRootFold = os.path.dirname(os.path.abspath(__file__))

    # bench data save-path
    BenchmarkDataSavePath_cold_run=os.path.join(ProjectRootFold, "Benchmark/timecost/data-cold_run.json")
    BenchmarkDataSavePath_hot_run=os.path.join(ProjectRootFold, "Benchmark/timecost/data-hot_run.json")
    BenchmarkDataAnalyzeSaveFold=os.path.join(ProjectRootFold, "Benchmark/images/")

    # onnx-model save-path
    OnnxSaveFold = os.path.join(ProjectRootFold, "Onnxs")
    TestDataCount = 10 


    @staticmethod
    def ChildModelSumParamsDict(model_name)->Dict[int,Dict[str,List[dict]]]:
        '''
        return convert "$project_path/RunLib/$target/$name/childs/$name-params.json" to dict
        '''

        if not os.path.exists(Config.ChildModelSumParamsSavePathName(model_name)):
            return None
        
        with open(Config.ChildModelSumParamsSavePathName(model_name),"r") as fp:
            try:
                return json.load(fp)
            except Exception as ex:
                print("error:",ex)
                return None

    def LoadModelParamsDictById(model_name,idx=-1)->Dict[str,List[dict]]:
        json_path=None

        if idx<0:
            json_path=Config.ModelParamsSavePathName(model_name)
        else:
            _,json_path=Config.ChildModelSavePathName(model_name,idx)

        if not os.path.exists(json_path):
            return None
        
        with open(json_path,"r") as fp:
            try:
                return json.load(fp)
            except Exception as ex:
                print("error:",ex)
                return None

    @staticmethod
    def ModelSavePathName(name) -> str:
        '''
        name is given when you create the data. Return "$project_path/Onnxs/$name/$name.onnx"
        '''
        os.makedirs(os.path.join(Config.OnnxSaveFold, name),exist_ok=True)
        return os.path.join(Config.OnnxSaveFold, name, name+".onnx")

    def ModelParamsSavePathName(name) -> str:
        '''
        name is given when you create the data. Return "$project_path/Onnxs/$name/$name.json"
        '''
        os.makedirs(os.path.join(Config.OnnxSaveFold, name),exist_ok=True)
        return os.path.join(Config.OnnxSaveFold, name, name+".json")

    @staticmethod
    def ChildModelSavePathName(name,idx) -> Tuple[str,str]:
        '''
        name is given when you create the data. Return "$project_path/Onnxs/$name/childs/idx/$name.onnx", "$project_path/Onnxs/$name/childs/idx/$name-$idx-params.json"
        '''
        
        os.makedirs(os.path.join(Config.OnnxSaveFold, name,"childs",str(idx)),exist_ok=True)
        return os.path.join(Config.OnnxSaveFold, name,"childs",str(idx), "{}-{}.onnx".format(name,str(idx))),os.path.join(Config.OnnxSaveFold, name,"childs",str(idx), "{}-{}-params.json".format(name,str(idx)))

    @staticmethod
    def ChildModelSumParamsSavePathName(name) -> str:
        '''
        name is given when you create the data. Return "$project_path/Onnxs/$name/childs/$name-params.json"
        '''
        
        os.makedirs(os.path.join(Config.OnnxSaveFold, name,"childs"),exist_ok=True)
        return os.path.join(Config.OnnxSaveFold, name,"childs", "{}-params.json".format(name))