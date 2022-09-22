import onnx
from onnx.onnx_ml_pb2 import NodeProto
from typing import List
from config import Config
import json
import os

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

        for input_name in list(node.input):
            if input_name in TotalParams:
                self.params.add(input_name)
            else:
                self.inputs.append(input_name)

        # idx in raw-model, start with idx=0
        self.idx=index                  # type: int

    def __str__(self) -> str:
        return "id={}, name={}, inputs={}, outputs={}, dependencies_inputs={}, dependencies_outputs={}".format(self.idx,self.name,self.inputs,self.outputs, self.dependencies_inputs, self.dependencies_outputs)

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

        self.nodes=[]           # type: list[GraphNode]
        self.params=None        # type: list[str]
        # self.shapes={}          # type: dict[str, dict[str, any]]        # {"name": {"shape": (-1,15), "type": "float32"}}
        
        if not self.Init():
            pass

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

    def SplitAndStoreChilds(self,childs: List[GraphNode]):
        '''
        split and store child onnx-models to disk with childs as start node. if real start not in, add we will add it automatically.
        '''
        
        childs=sorted(childs,key=lambda x: x.idx)
        if len(childs)<1 or childs[0].idx!=0:
            childs.insert(0,self.nodes[0])

        for child_idx in range(len(childs)):
            start_node=childs[child_idx]
            end_node=self.nodes[-1]
            if child_idx+1<len(childs):
                end_node=self.nodes[childs[child_idx+1].idx-1]

            print("{}-{} ==|>".format(self.modelName,child_idx), start_node.dependencies_inputs,"-->", end_node.dependencies_outputs)

            # params=set()
            # for idx in range(start_node.idx,end_node.idx+1):
            #     params |= self.nodes[idx].params

            # print(params)

            child_onnx_path,child_params_path=Config.ChildModelSavePathName(self.modelName,child_idx)
            try:
                onnx.utils.extract_model(self.onnxPath,child_onnx_path , start_node.dependencies_inputs, end_node.dependencies_outputs)
            except onnx.onnx_cpp2py_export.checker.ValidationError:
                params=set()
                for idx in range(start_node.idx,end_node.idx+1):
                    params |= self.nodes[idx].params
                onnx.utils.extract_model(self.onnxPath,child_onnx_path , start_node.dependencies_inputs+list(params), end_node.dependencies_outputs)
            except Exception as ex:
                print("error{}: ".format(type(ex)),ex)
                return 
            ModelAnalyzer.CreateParamsInfo(child_onnx_path,child_params_path)

    @staticmethod
    def CreateParamsInfo(onnx_path:str,params_path:str)->bool:
        model = onnx.load(onnx_path)
        
        params_dict={"input": [], "output":[]}

        weight_params=set([v.name for v in model.graph.initializer])

        for k, tennsors in {"input": model.graph.input, "output": model.graph.output}.items():
            for m in tennsors:
                param={}

                if str(m.name) in weight_params:
                    continue

                param["type"] = OnnxType.GetElemType(m.type.tensor_type.elem_type)
                param["name"] = str(m.name)
                shape_list=[]
                for v in m.type.tensor_type.shape.dim:
                    shape_list.append(v.dim_value if isinstance(v.dim_value, int) and v.dim_value>0  else -1)
                param["shape"] = tuple(shape_list)
                params_dict[k].append(param)

            os.makedirs(os.path.dirname(os.path.abspath(params_path)),exist_ok=True)
            with open(params_path,"w") as fp:
                json.dump(params_dict,fp,indent=4)

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