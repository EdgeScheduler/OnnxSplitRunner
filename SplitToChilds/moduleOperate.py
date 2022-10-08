from operator import index
from selectors import EpollSelector
import onnx
from onnx.onnx_ml_pb2 import NodeProto
from typing import List
from config import Config
import json
import os
import time
from typing import Dict,List

class OnnxType:
    @staticmethod
    def GetElemType(id:int)->str:
        if id==1:
            return "float32"
        elif id==2:
            return "uint8"
        elif id==3:
            return "int8"
        elif id==4:
            return "uint16"
        elif id==5:
            return "int16"
        elif id==6:
            return "int32"
        elif id==7:
            return "int64"
        elif id==8:
            return "string"
        elif id==9:
            return "boolean"
        elif id==10:
            return "float16"
        elif id==11:
            return "float64"
        elif id==12:
            return "uint32"
        elif id==14:
            return "uint64"
        elif id==15:
            return "complex128"
        elif id==16:
            return "bfloat16"

class GraphNode():
    def __init__(self,node: NodeProto, TotalParams: List[str]=[],index:int=-1):
        self.name=str(node.name)        # type: str
        self.type=str(node.op_type)

        # inputs of current node
        self.inputs=[]                  # type: list[str]

        # outputs of current node                                                          
        self.outputs=list(node.output)  # type: list[str]
        
        # dependencies inputs of nodes that idx >= self.idx
        self.dependencies_inputs=[]     # type: list[str]

        # dependencies outputs of nodes that idx >= self.idx
        self.dependencies_outputs=[]    # type: list[str]

        self.params=set()               # type: set[str]

        self.input_info={}              # {"data": [{"type": str, "name": str, "shape": list, "cost": float}], "cost": float}
        # self.output_info={}           # {"data": [{"type": str, "name": str, "shape": list, "cost": float}], "cost": float}

        for input_name in list(node.input):
            if input_name in TotalParams:
                self.params.add(input_name)
            else:
                self.inputs.append(input_name)

        # idx in raw-model, start with idx=0
        self.idx=index                  # type: int

    def __str__(self) -> str:
        return "id={}, name={}, inputs={}, outputs={}, dependencies_inputs={}, dependencies_outputs={}, input_info: {}".format(self.idx,self.name,self.inputs,self.outputs, self.dependencies_inputs, self.dependencies_outputs, self.input_info)

    def IsConvergeNode(self)->bool:
        return True if len(self.dependencies_inputs)<2 else False

# enable: for node in model_analyzer: ...
class ModelAnalyzerIterator():
    def __init__(self,nodes) -> None:
        self.items=nodes
        self.index=0
    
    def __next__(self):
        if self.index<len(self.items):
            self.index+=1
            return self.items[self.index-1]
        else:
            raise StopIteration

class ModelAnalyzer():
    def __init__(self,model_name:str,onnx_path:str=None):
        self.modelName=model_name

        if onnx_path is None:
            self.onnxPath=Config.ModelSavePathName(model_name)
        else:
            self.onnxPath=onnx_path

        self.nodes=[]                   # type: list[GraphNode]
        self.start_node=None            # type: GraphNode                       # find the first node that enable to be end. if raw_input in output, it will get 'KeyError', we need this value until we find a solution
        self.params=None                # type: list[str]
        # self.shapes={}                # type: dict[str, dict[str, any]]       # {"name": {"shape": (-1,15), "type": "float32"}}

        self.use_cache=True
        
        if not self.Init():
            return

        self.RuntimeAnalyze()

    def Init(self)->bool:
        try:
            model = onnx.load(self.onnxPath)
            self.params= set([v.name for v in model.graph.initializer])

            # params shape
            # for v in model.graph.initializer:
            #     if v not in self.shapes:
            #         self.shapes[v]={}
            #         self.shapes[v]["shape"]=tuple(v.dims)
            #         self.shapes[v]["type"]=OnnxType.GetElemType(v.data_type)

            # for v in model.input:
            #     pass

            # for v in model.output:
            #     pass

            # print(model)
            for idx,node in enumerate(model.graph.node):
                self.nodes.append(GraphNode(node=node,TotalParams=self.params,index=idx))

            self.RecordDependency()
        except Exception as ex:
            print("error: fail to init model-analyzer")
            print(str(ex))
            return False
        return True

    def SetEnableCache(self,enable:bool=True):
        self.use_cache=enable

    def EnableStart(self,node:GraphNode)->bool:
        if node==self.nodes[0] or node.idx>self.start_node.idx:
            return True
        else:
            return False

    def LoadCache(self)->dict:
        return Config.LoadChildModelSumCacheDict(self.modelName)

    def RuntimeAnalyze(self):
        print("start to analyze model, it may cost some time.")
        cache={}
        if self.use_cache:
            cache=Config.LoadChildModelSumCacheDict(self.modelName)

        if len(cache)<1 or "data" not in cache or len(cache["data"])!=len(self.nodes):
            cache["model-path"]=self.onnxPath
            cache["model-name"]=self.modelName
            cache["create-time"]=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cache["data"]={}
            tmp_onnx_path,tmp_param_path=Config.TempChildModelSavePathName(self.modelName)
            for idx,node in enumerate(self.nodes):
                info = self.ExtractModelByNode(self.onnxPath,tmp_onnx_path,tmp_param_path,node,node,print_error=False)
                
                if info is not None:
                    node.input_info=info["input"]
                    # node.output_info=info["output"]
                    if self.start_node is None:
                        self.start_node=node
                cache["data"][str(idx)]=info
            Config.RemoveTempChildModelSavePathName(self.modelName)
        else:
            print("load cache created at %s"%(cache["create-time"]))
            for idx,node in enumerate(self.nodes):
                info=cache["data"][str(idx)]
                if info is not None:
                    if "input" in info:
                        node.input_info=info["input"]
                    # if "output" in info:
                    #     node.output_info=info["output"]
                    if self.start_node is None and ("global" not in info or not info["global"]):
                        self.start_node=node

        if len(self.nodes[0].input_info)<1:
            info=ModelAnalyzer.CreateParamsInfo(self.onnxPath,Config.ModelParamsSavePathName(self.modelName))
            self.nodes[0].input_info=info["input"]
            cache["data"]["0"]=info
            cache["global"]=True
            if "output" in cache["data"]["0"]:
                del cache["data"]["0"]["output"]

        with open(Config.ChildModelSumCacheSavePathName(self.modelName),"w") as fp:
            json.dump(cache,fp,indent=4)
        print("model analyze ok.")

    def ExtractModelByNode(self,raw_onnx_path:str,new_onnx_path:str,new_onnx_praram_path:str,start_node: GraphNode,end_node: GraphNode, print_error=True)->Dict[str,dict]:
        try:
            onnx.utils.extract_model(raw_onnx_path,new_onnx_path, start_node.dependencies_inputs, end_node.dependencies_outputs)
        except onnx.onnx_cpp2py_export.checker.ValidationError:
            params=set()
            for idx in range(start_node.idx,end_node.idx+1):
                params |= self.nodes[idx].params
            onnx.utils.extract_model(raw_onnx_path,new_onnx_path , start_node.dependencies_inputs+list(params), end_node.dependencies_outputs)
        except Exception as ex:
            # print(raw_onnx_path,new_onnx_path , start_node.dependencies_inputs, end_node.dependencies_outputs)
            if print_error:
                print("error {}: ".format(type(ex)),ex)
            return None

        return ModelAnalyzer.CreateParamsInfo(new_onnx_path,new_onnx_praram_path)

    # don't consider extra-output in middle of model at this time
    def RecordDependency(self):
        dependency=set()
        for idx in range(len(self.nodes))[::-1]:
            if idx==len(self.nodes)-1:
                for out in self.nodes[idx].outputs:
                    if out not in self.params:
                        self.nodes[idx].dependencies_outputs.append(out)

            for input_name in self.nodes[idx].inputs:
                dependency.add(input_name)

            for output_name in self.nodes[idx].outputs:
                # if output_name in dependency:
                dependency.discard(output_name)

            self.nodes[idx].dependencies_inputs=list(dependency)

            if idx>0:
                # out=set(self.nodes[idx].dependencies_inputs) | set(self.nodes[idx-1].outputs)
                # self.nodes[idx-1].dependencies_outputs=list(out)
                self.nodes[idx-1].dependencies_outputs=self.nodes[idx].dependencies_inputs

    def SplitAndStoreChilds(self,childs: List[GraphNode])->dict:
        '''
        split and store child onnx-models to disk with childs as start node. if real start not in, add we will add it automatically.
        '''
        
        total_param={}
        childs=sorted([child for child in childs if self.EnableStart(child)],key=lambda x: x.idx)
        if len(childs)<1 or childs[0].idx!=0:
            childs.insert(0,self.nodes[0])

        # delete repeated item
        childs_=[]
        tmp=-1
        for child in childs:
            if child.idx>tmp:
                childs_.append(child)
                tmp=child.idx
        childs=childs_

        info=ModelAnalyzer.CreateParamsInfo(self.onnxPath,Config.ModelParamsSavePathName(self.modelName))
        info["from"]=self.nodes[0].idx
        info["to"]=self.nodes[-1].idx
        total_param[-1]=info

        for child_idx in range(len(childs)):
            start_node=childs[child_idx]
            end_node=self.nodes[-1]
            if child_idx+1<len(childs):
                # print("debug: ",[v.idx for v in childs],childs[child_idx+1].idx-1)
                end_node=self.nodes[childs[child_idx+1].idx-1]

            print("{}-{} ==|>".format(self.modelName,child_idx), start_node.name,"-->", end_node.name)

            # params=set()
            # for idx in range(start_node.idx,end_node.idx+1):
            #     params |= self.nodes[idx].params

            # print(params)

            child_onnx_path,child_params_path=Config.ChildModelSavePathName(self.modelName,child_idx)
            info = self.ExtractModelByNode(self.onnxPath,child_onnx_path,child_params_path,start_node,end_node)
            info["from"]=start_node.idx
            info["to"]=end_node.idx
            total_param[child_idx]=info

            # try:
            #     onnx.utils.extract_model(self.onnxPath,child_onnx_path , start_node.dependencies_inputs, end_node.dependencies_outputs)
            # except onnx.onnx_cpp2py_export.checker.ValidationError:
            #     params=set()
            #     for idx in range(start_node.idx,end_node.idx+1):
            #         params |= self.nodes[idx].params
            #     onnx.utils.extract_model(self.onnxPath,child_onnx_path , start_node.dependencies_inputs+list(params), end_node.dependencies_outputs)
            # except Exception as ex:
            #     print("error{}: ".format(type(ex)),ex)
            #     return 
            # total_param[child_idx] = ModelAnalyzer.CreateParamsInfo(child_onnx_path,child_params_path)

        with open(Config.ChildModelSumParamsSavePathName(self.modelName),"w") as fp:
            json.dump(total_param,fp,indent=4)

        return total_param

    @staticmethod
    def CreateParamsInfo(onnx_path:str,params_path:str,default_batch=15)->Dict[str,dict]:
        model = onnx.load(onnx_path)
        
        params_dict={"input": {"data":[], "cost":0}, "output":{"data":[], "cost":0}, "model_path": onnx_path}

        weight_params=set([v.name for v in model.graph.initializer])

        for k, tennsors in {"input": model.graph.input, "output": model.graph.output}.items():
            for m in tennsors:
                param={}

                if str(m.name) in weight_params:
                    continue

                param["type"] = OnnxType.GetElemType(m.type.tensor_type.elem_type)
                param["name"] = str(m.name)
                shape_list=[]
                mul_value=1
                for v in m.type.tensor_type.shape.dim:
                    shape_list.append(v.dim_value if isinstance(v.dim_value, int) and v.dim_value>0  else -1)
                    mul_value*=v.dim_value if isinstance(v.dim_value, int) and v.dim_value>0  else default_batch
                param["shape"] = tuple(shape_list)
                
                cost=0.0
                if "int" in param["type"]:
                    cost=mul_value*4
                else:
                    cost=mul_value*4
                param["cost"]=cost/(1024*1024)                                              # MB
                params_dict[k]["cost"]+=cost
                params_dict[k]["data"].append(param)
            params_dict[k]["cost"]=params_dict[k]["cost"]/(1024*1024)                       # MB

            os.makedirs(os.path.dirname(os.path.abspath(params_path)),exist_ok=True)
            with open(params_path,"w") as fp:
                json.dump(params_dict,fp,indent=4)
        return params_dict

    def GetConvergeNodes(self)->List[GraphNode]:
        result=[]
        for node in self.nodes:
            if node.IsConvergeNode():
                result.append(node)
        return result

    def GetAllNodes(self)->List[GraphNode]:
        return self.nodes

    def __str__(self) -> str:
        return "".join([str(node)+"\n" for node in self.nodes])

    def __getitem__(self,index)->GraphNode:
        return self.nodes[index]

    def __len__(self)->int:
        return len(self.nodes)

    def __iter__(self):
        return ModelAnalyzerIterator(self.nodes)