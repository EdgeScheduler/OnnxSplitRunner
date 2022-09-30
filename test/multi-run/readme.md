how to run: 

```shell
docker pull mcr.microsoft.com/azureml/onnxruntime:latest-cuda
docker run -it --name=test-myonnx --runtime=nvidia  -v /home/onceas/yutian/OnnxSplitRunner:/mytest  mcr.microsoft.com/azureml/onnxruntime:latest-cuda /bin/bash

cd /mytest
python3 test/multi-run/googlenet.py
```