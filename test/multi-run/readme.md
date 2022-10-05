# how to run onnxruntime by docker: 

```shell
docker pull mcr.microsoft.com/azureml/onnxruntime:latest-cuda
docker run -it --name=test-myonnx --runtime=nvidia  -v /home/onceas/yutian/OnnxSplitRunner:/mytest  mcr.microsoft.com/azureml/onnxruntime:latest-cuda /bin/bash

cd /mytest
python3 test/multi-run/googlenet.py
```

# test with k8s
test environment is `GPU: tesla-T4`, `inference Batch size`=15, count=900


| model-name | serial-run(s) | multi-process(s) | pods-gpu-manager(s) |
| :---: | :---: | :---: | :---: |
| googlenet | 16.45 | 75.21 | 72.68 |
| resnet50 | 31.71 | 83.91 | 171.74 |
| squeezenetv1 | 11.44 | 59.20 | 58.35 |
|vgg19 | 96.14 | 152.28 | 379.66 |

``` yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-muti-process-with-onnx-gpu
  namespace: default
spec:
  replicas: 4
  selector:
    matchLabels:
      app: test-muti-process
  template:
    metadata:
      labels:
        app: test-muti-process
      
    spec:
      containers:
      - name: onnxruntime-gpu
        image: mcr.microsoft.com/azureml/onnxruntime:latest-cuda
        volumeMounts:
        - mountPath: /my-test
          name: code-volume
        command: ["/bin/sh"]
        args:
          [
            "-c",
            "sleep inf",
          ]
        imagePullPolicy: IfNotPresent
        resources:
          requests:
            tencent.com/vcuda-core: 25
            tencent.com/vcuda-memory: 10
          limits:
            tencent.com/vcuda-core: 25
            tencent.com/vcuda-memory: 10
      nodeSelector:
        kubernetes.io/hostname: dell02
      volumes:
      - name: code-volume
        hostPath:
          path: /OnnxSplitRunner
          type: DirectoryOrCreate
```