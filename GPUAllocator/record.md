Tesla-T4: gpu-manager all with 25%
| model-name | serial-run(s) | multi-process(s) | pods-gpu-manager(s) |
| :---: | :---: | :---: | :---: |
| googlenet | 0.018 | 0.084 | 0.081 |
| resnet50 | 0.035 | 0.093 | 0.191 |
| squeezenetv1 | 0.0127 | 0.066 | 0.065 |
|vgg19 | 0.107 | 0.169 | 0.422 |

GTX-2080Ti:
| model-name | serial-run(s) | multi-process(s) | threads(s) |
| :---: | :---: | :---: | :---: | 
| googlenet | 0.009| 0.046 | 0.066 | 
| resnet50 | 0.015 | 0.032 | 0.065 |
| squeezenetv1 | 0.006 | 0.041 | 0.064 |
| vgg19 | 0.039 | 0.069 | 0.058 |

child-model with 2080Ti:
| model-name | 0(s) | 1(s) | 2(s) |
| :---: | :---: | :---: | :---: |
| googlenet | 0.007 | - | - | - |
| resnet50 | 0.011  | 0.006 | - | 
| squeezenetv1 | 0.005  | - | - | 
| vgg19 | 0.016  | 0.016 | 0.019 |

