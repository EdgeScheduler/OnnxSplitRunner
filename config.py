import os
from typing import Dict,List
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
    TVMLibSaveFold = os.path.join(ProjectRootFold, "RunLib")

    # model-functions text save-path
    RawModelFunctionsTextSaveFold = os.path.join(ProjectRootFold, "ModelFuntionsText/raw")
    ChildsModelFunctionsTextSaveFold= os.path.join(ProjectRootFold, "ModelFuntionsText/childs")

    # model-functions Python-file save-path
    RawModelFunctionsPythonSaveFold = os.path.join(ProjectRootFold, "ModelFuntionsPython/raw")
    ChildsModelFunctionsPythonSaveFold= os.path.join(ProjectRootFold, "ModelFuntionsPython/childs")

    TestDataCount = 10 

    @staticmethod
    def RawModelFunctionsTextSavePathName(model_name)->str:
        '''
        name is given when you use raw model functions text from disk, you may create this file by print-copy. Return "$project_path/ModelFuntionsText/raw/$model_name.txt"
        '''
        return os.path.join(Config.RawModelFunctionsTextSaveFold,model_name+".txt")

    @staticmethod
    def RawModelFunctionsPythonSavePathName(model_name)->str:
        '''
        name is given when you use raw model functions Python-file from disk, you may create this file by print-copy. Return "$project_path/ModelFuntionsPython/raw/$model_name.py"
        '''
        return os.path.join(Config.RawModelFunctionsPythonSaveFold,model_name+".py")

    @staticmethod
    def ModelParamsFile(model_name)->Dict[int,List[dict]]:
        '''
        return convert "$project_path/ModelFuntionsText/childs/$model_name/params.json" to dict
        '''
        jsonFilePath=os.path.join(Config.ChildModelFunctionsTextSaveFold(model_name),"params.json")
        if not os.path.exists(jsonFilePath):
            return None
        
        with open(jsonFilePath,"r") as fp:
            try:
                return json.load(fp)
            except Exception as ex:
                print("error:",ex)
                return None
    
    @staticmethod
    def ChildModelFunctionsPythonSavePathName(model_name)->str:
        '''
        name is given when you use child-model functions Python-file from disk, you may create this file by print-copy. Return "$project_path/ModelFuntionsPython/childs/$model_name.py"
        '''
        return os.path.join(Config.ChildsModelFunctionsPythonSaveFold,model_name+".py")

    @staticmethod
    def ChildModelFunctionsTextSaveFold(model_name)->str:
        '''
        name is given when you use child-model functions Text-file fold from disk, you may create this file by print-copy. Return "$project_path/ModelFuntionsText/childs/$model_name/"
        '''
        return os.path.join(Config.ChildsModelFunctionsTextSaveFold,model_name)


    @staticmethod
    def ModelSavePathName(name) -> str:
        '''
        name is given when you create the data. Return "$project_path/Onnxs/$name/$name.onnx"
        '''
        os.makedirs(os.path.join(Config.OnnxSaveFold, name),exist_ok=True)
        return os.path.join(Config.OnnxSaveFold, name, name+".onnx")

    @staticmethod
    def ChildModelSavePathName(name,idx) -> str:
        '''
        name is given when you create the data. Return "$project_path/Onnxs/$name/childs/idx/$name.onnx", "$project_path/RunLib/$target/$name/$idx/$name-$idx-params.json"
        '''
        
        os.makedirs(os.path.join(Config.OnnxSaveFold, name,"childs",str(idx)),exist_ok=True)
        return os.path.join(Config.OnnxSaveFold, name,"childs",str(idx), "{}-{}.onnx".format(name,str(idx))),os.path.join(Config.OnnxSaveFold, name,"childs",str(idx), "{}-{}-params.json".format(name,str(idx)))

    @staticmethod
    def TvmLibSavePathByName(name, target, idx: int=-1) -> str:
        '''
        name is given when you create the data. Return "$project_path/RunLib/$target/$name/$name-$idx.tar", "$project_path/RunLib/$target/$name/$idx/$name-$idx-input_shape.json"
        '''
        
        if idx>=0:
            fold=os.path.join(Config.TVMLibSaveFold,target,name,"childs",str(idx))
            os.makedirs(fold,exist_ok=True)

            return os.path.join(fold, "{}-{}.tar".format(name,str(idx))),os.path.join(fold, "{}-{}-input_shape.json".format(name,str(idx)))
        else:
            fold=os.path.join(Config.TVMLibSaveFold,target,name,"raw")
            os.makedirs(fold,exist_ok=True)

            return os.path.join(fold, "{}.tar".format(name)),os.path.join(fold, "{}-input_shape.json".format(name))

    @staticmethod
    def ModelSaveDataPathName(name) -> str:
        '''
        name is given when you create the data. Return "$project_path/Onnxs/$name/data.json"
        '''
        return os.path.join(Config.OnnxSaveFold, name, "data.json")