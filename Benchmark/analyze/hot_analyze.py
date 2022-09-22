import matplotlib.pyplot as plt
import json
import os
from config import Config
import numpy as np

def mean(l:list)->float:
    return sum(l)/len(l)

def mycolor(index)->str:
    colors_ = ['red','green','blue','c','m','y','k']

    if index<len(colors_) and index>=0:
        return colors_[index]
    else:
        return 'w'

data={}
with open(Config.BenchmarkDataSavePath_hot_run, "r") as fp:
    data = json.load(fp)    # type: dict[str,dict[str,dict]]

# 开始画图
# plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False      # 解决保存图像时'-'显示为方块的问题
picture_index=1

for model_name,model_value in data.items():
    device_count=0
    # 开始绘图
    plt.figure(picture_index,figsize=(100,25))
    plt.suptitle(model_name,fontsize = 40)
    for device,device_value in model_value.items():
        count=device_value["count"]

        onnx_data=device_value["whole_time_by_onnx"]
        function_data=device_value["whole_time_by_function"]
        child_data_whole=device_value["whole_time_by_child"]["whole"]
        child_data_split=device_value["whole_time_by_child"]["childs"]

        onnx_mem_data=device_value["whole_gpu_memory_by_onnx"]
        function_mem_data=device_value["whole_gpu_memory_by_function"]
        child_mem_data_whole=device_value["whole_gpu_memory_by_childs"]["whole"]
        child_mem_data_split=device_value["whole_gpu_memory_by_childs"]["childs"]

        # 计算累加
        # start=0
        # result=[]
        # for change in child_mem_data_split:
        #     start+=change
        #     result.append(start)
        # child_mem_data_split=result
        # child_mem_data_whole=max(result)

        # 创建存储目录
        os.makedirs(os.path.join(Config.BenchmarkDataAnalyzeSaveFold,"hot_run/"),exist_ok=True)

        # 绘制执行时间总比图
        img = plt.subplot(2, 4 , 1+4*device_count)
        x_labels=["run by onnx","run by expr-function","run by split-childs"]
        x=np.arange(len(x_labels))
        plt.barh(x,[onnx_data*1000.0,function_data*1000.0,child_data_whole*1000.0],0.2)

        plt.xlabel('hot-run time-cost(ms)')
        plt.yticks(x)
        img.set_yticklabels(x_labels)
        img.set_title("{}-hot-runtime-cost".format(device))


        # 绘制子模型变化图
        img = plt.subplot(2,4, 2+4*device_count)
        x=range(1,len(child_data_split)+1)
        plt.plot(x,[v*1000.0 for v in child_data_split],marker="o")
        plt.xticks(x)
        plt.xlabel('child-idx')
        plt.ylabel('hot-run time-cost(ms)')
        img.set_title("{}-child-model runtime-cost change".format(device))

        plt.savefig(os.path.join(Config.BenchmarkDataAnalyzeSaveFold,"hot_run/",model_name+".png"))


        # 绘制执行显存总比图
        img = plt.subplot(2, 4 , 3+4*device_count)
        x_labels=["run by onnx","run by expr-function","run by split-childs"]
        x=np.arange(len(x_labels))
        plt.barh(x,[onnx_mem_data,function_mem_data,child_mem_data_whole],0.2)

        plt.xlabel('hot-run gpu-memory-cost(MB)')
        plt.yticks(x)
        img.set_yticklabels(x_labels)
        img.set_title("{}-hot-gpu-memory-cost".format(device))

        plt.savefig(os.path.join(Config.BenchmarkDataAnalyzeSaveFold,"hot_run/",model_name+".png"))

        # 绘制子模型显存变化图
        img = plt.subplot(2,4, 4+4*device_count)
        x=range(1,len(child_mem_data_split)+1)
        plt.plot(x,child_mem_data_split,marker="o")
        plt.xticks(x)
        plt.xlabel('child-idx')
        plt.ylabel('hot-run gpu-memory-cost(MB)')
        img.set_title("{}-child-model gpu-memory-cost change".format(device))

        plt.savefig(os.path.join(Config.BenchmarkDataAnalyzeSaveFold,"hot_run/",model_name+".png"))

        # end
        device_count+=1
    picture_index += 1