旨在设计一个可能真实存在的任务场景，通过实验表明通过模型划分能够有效提升单任务的推理效率。
> 以下所有实验，默认推理的Batch-size=15，测试包含`resnet50`、`googlenet`、`squeezenetv1`、`vgg19`.

假定起始时刻t0=0， 任务以如下方式的到达（真实情况：任务持续随机到达）：
> 子模型数量：
> VGG19: 4
> squeezenetv1: 1
> googlet: 1
> resnet50: 2

1. vgg19
2. squeezenetv1
3. googlenet
4. resnet50
5. vgg19
6. googlenet
