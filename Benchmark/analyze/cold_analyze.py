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
with open(Config.BenchmarkDataSavePath_cold_run, "r") as fp:
    data = json.load(fp)    # type: dict[str,dict[str,dict]]

# 开始画图
# plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False      # 解决保存图像时'-'显示为方块的问题
picture_index=1

for model_name,model_value in data.items():
    device_count=0
    # 开始绘图
    plt.figure(picture_index,figsize=(60,25))
    plt.suptitle(model_name,fontsize = 40)
    for device,device_value in model_value.items():
        count=device_value["count"]
        onnx_data=device_value["whole_time_by_onnx"]
        function_data=device_value["whole_time_by_function"]
        child_data_whole=[value["whole"] for value in  device_value["whole_time_by_child"]]
        child_data_splits=[value["childs"] for value in  device_value["whole_time_by_child"]]

        # 创建存储目录
        os.makedirs(os.path.join(Config.BenchmarkDataAnalyzeSaveFold,"cold_run/"),exist_ok=True)

        # 绘制执行时间总比图
        img = plt.subplot(2,3 , 1+3*device_count)
        x_labels=["max","min","mean","first","mean[1:]"]
        x=np.arange(len(x_labels))
        y_onnx=[max(onnx_data),min(onnx_data),mean(onnx_data),onnx_data[0],mean(onnx_data[1:])]
        y_function=[max(function_data),min(function_data),mean(function_data),function_data[0],mean(function_data[1:])]
        y_childs=[max(child_data_whole),min(child_data_whole),mean(child_data_whole),child_data_whole[0],mean(child_data_whole[1:])]
        bar_width=0.2
    
        plt.barh(x-bar_width,y_onnx,bar_width, color="r",label="run by onnx")
        plt.barh(x,y_function,bar_width,color="g",label="run by expr-function")
        plt.barh(x+bar_width,y_childs,bar_width,color="blue",label="run by split-childs")
        plt.legend() # 显示图例
        img.set_yticks(x)
        img.set_yticklabels(x_labels)
        plt.xlabel('cold-run time-cost(s)')
        img.set_title("{}-cold-runtime-cost statistics".format(device))

        # 绘制执行时间总比图
        img = plt.subplot(2,3 , 2+3*device_count) 
        x=range(1,count+1)

        plt.plot(x,onnx_data,color="r",label="run by onnx")
        plt.plot(x,function_data,color="g",label="run by expr-function")
        plt.plot(x,child_data_whole,color="blue",label="run by split-childs")
        plt.legend() # 显示图例
        plt.xticks(x)
        plt.xlabel('x-th round')
        plt.ylabel('cold-run time-cost(s)')
        img.set_title("{}-cold-runtime-cost change".format(device))

        # 绘制子模型变化图
        img = plt.subplot(2,3 , 3+3*device_count)
        x=range(1,len(child_data_splits[0])+1)

        for index,child_data_split in enumerate(child_data_splits):
            plt.plot(x,child_data_split,marker="o",color=mycolor(index),label="{}th time run".format(index+1))
        plt.legend() # 显示图例
        plt.xticks(x)
        plt.xlabel('child-idx')
        plt.ylabel('cold-run time-cost(s)')
        img.set_title("{}-child-model runtime-cost change".format(device))

        plt.savefig(os.path.join(Config.BenchmarkDataAnalyzeSaveFold,"cold_run/",model_name+".png"))

        # end
        device_count+=1
    picture_index += 1