# 函数功能
```
|-- pointnet_cls.py       #点云分类
|-- pointnet_part_seg.py  #点云部件分割
|-- pointnet_sem_seg.py   #点云场景分割
```

# 模型可视化
- 测试一
```
from pointnet_utils import PointNetEncoder
from torchsummary import summary

def main():
    model = PointNetEncoder().cuda()
    summary(model, (3, 1024))

if __name__ == '__main__':
    main()
```

```
#输出
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1             [-1, 64, 1024]             256
       BatchNorm1d-2             [-1, 64, 1024]             128
            Conv1d-3            [-1, 128, 1024]           8,320
       BatchNorm1d-4            [-1, 128, 1024]             256
            Conv1d-5           [-1, 1024, 1024]         132,096
       BatchNorm1d-6           [-1, 1024, 1024]           2,048
            Linear-7                  [-1, 512]         524,800
       BatchNorm1d-8                  [-1, 512]           1,024
            Linear-9                  [-1, 256]         131,328
      BatchNorm1d-10                  [-1, 256]             512
           Linear-11                    [-1, 9]           2,313
            STN3d-12                 [-1, 3, 3]               0
           Conv1d-13             [-1, 64, 1024]             256
      BatchNorm1d-14             [-1, 64, 1024]             128
           Conv1d-15             [-1, 64, 1024]           4,160
      BatchNorm1d-16             [-1, 64, 1024]             128
           Conv1d-17            [-1, 128, 1024]           8,320
      BatchNorm1d-18            [-1, 128, 1024]             256
           Conv1d-19           [-1, 1024, 1024]         132,096
      BatchNorm1d-20           [-1, 1024, 1024]           2,048
           Linear-21                  [-1, 512]         524,800
      BatchNorm1d-22                  [-1, 512]           1,024
           Linear-23                  [-1, 256]         131,328
      BatchNorm1d-24                  [-1, 256]             512
           Linear-25                 [-1, 4096]       1,052,672
            STNkd-26               [-1, 64, 64]               0
           Conv1d-27            [-1, 128, 1024]           8,320
      BatchNorm1d-28            [-1, 128, 1024]             256
           Conv1d-29           [-1, 1024, 1024]         132,096
      BatchNorm1d-30           [-1, 1024, 1024]           2,048
================================================================
Total params: 2,803,529
Trainable params: 2,803,529
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 57.09
Params size (MB): 10.69
Estimated Total Size (MB): 67.79
----------------------------------------------------------------

进程已结束,退出代码0

```
- 测试二
```
import torch
from pointnet_cls import get_model

def main():
    ins = torch.randn([16, 6, 1024]).cuda()
    model = get_model().cuda()
    out1, out2 = model(ins)
    print(out1.shape, out2.shape)

def __name__ == "__main__":
    main()
```

```
#输出
torch.Size([4, 40]) torch.Size([4, 64, 64])

进程已结束,退出代码0
```
