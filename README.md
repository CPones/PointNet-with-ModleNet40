# PointNet-with-ModleNet40
基于深度学习的3D点云数据处理——开山之作PointNet模型【Pytorch版】

## 一、项目介绍
点云数据是一种三维坐标的数据集合，广泛存在于激光扫描、自动驾驶、增强现实等应用领域，其中包含丰富的目标轮廓信息，基于深度学习的3D点云数据处理已成为计算机视觉的重要组成部分。PointNet开创性的提出了一种新型深度神经网络结构，解决了点云数据的无序性、非结构性和置换不变性，能够应用在点云分类、点云分割和场景分割的视觉任务中。本项目使用Pytorch搭建模型训练环境，在ModelNet40训练集上的分类精度为91.9%。

### 1.1 PointNet模型
![](./images/PointNet.png)
- Classification Network 和 Segmentation Network 两个部分组成，分别应用于点云分类和点云分割任务。
- 输入维度`n×3`，输出维度`n×k`或`n×m`，其中`n`为点云个数，`m`为分类类别数，`k`为分割类别数。
- T-Net空间旋转变换，mlp多层感知机，max pool全局池化层，global feature特征融合。

### 1.2 数据集
![](./images/ModelNet40.png)
- 左图为三维模型类别词云：右图为chair三维模型可视化。
- 数据集下载地址：[ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)
- 类型名称列表：airplane，bathtub，bed，bench，bookshelf，bottle，bowl，car，chair，cone，cup，curtain，desk，door，dresser，flower_pot，glass_box，guitar，keyboard，lamp，laptop，mantel，monitor，night_stand，person，piano，plant，radio，range_hood，sink，sofa，stairs，stool，table，tent，toilet，tv_stand，vase，wardrobe，xbox。
```
./data/modelnet40_normal_resampled/
  |--airplane
    |--airplane_0001.txt
    |--airplane_0002.txt
    |--……
  |--bathtub
    |--bathtub_0001.txt
    |--bathtub_0002.txt
    |--……
  |--……
  |--filelist.txt #文件目录
  |--modelnet10_shape_name.txt  # modelnet10类别名
  |--modelnet10_train.txt       # modelnet10训练集
  |--modelnet10_test.txt        # modelnet10测试集
  |--modelnet40_shape_name.txt  # modelnet40类别名
  |--modelnet40_train.txt       # modelnet40训练集
  |--modelnet40_test.txt        # modelnet40测试集
```

### 1.3 数据增强
数据增强扩大训练样本数量，提升模型泛化能力，有效避免过拟合。选择合适的数据增强策略能提升模型精度，本项目提供几种常用点云数据增强函数，保存在`provider.py`文件中。
- normalize_data：归一化
- rotate_point_cloud：空间坐标旋转，α、β、γ轴
- jitter_pointnet_cloud：添加高斯噪声
- shift_pointnet_cloud：空间坐标平移，x、y、z轴
- random_scale_point_cloud：随机缩放
- random_point_dropout：随机正则化

## 二、项目运行
### 2.1 环境信息
- CPU:AMD RX3600 6core 3.60GHz 16G
- GPU:NIVIDIA GTX 1650 4G
- Windons10
- Python3.7
- pytorch1.7.1
- Cuda10.1
- cuDNN7.6.5
```
pip install -r requirements.txt
```

### 2.2 数据集准备
数据集下载路径[ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)，解压后存放在`data/modelnet40_normal_resampled`路径下。

### 2.2 模型训练
```
python train.py
```

### 2.3 模型测试
```
python test.py
```

### 2.4 结果保存
模型训练结果保存[地址](https://drive.google.com/file/d/1pWz0FylnNv5jn2jhuKHoHljBYDPKvuE6/view?usp=sharing)
| model | accuracy |
|  :-:  |   :-:    |
| pointnet(official)  | 89.2% |
| pointnet(ours39.8MB)| 91.9% |


## 三、项目总结
PointNet作为基于深度学习处理点云数据的开山之作，提供了一种全新的深度神经网络结构，具有开创新时代的意义。本项目使用Pytorch搭建模型训练环境，模型精度可以达到论文给出的89.2%，开源贡献者们可以尝试使用数据增强策略进一步提升精度。由于PointNet自身存在一定的局限性，后续的研究者相继对其进行模型改进，主要方式集中在局部特征融合、卷积核算子、注意力机制等方面。

## 四、参考引用
>开源项目：[https://github.com/yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

>论文下载：[https://arxiv.org/pdf/1612.00593.pdf](https://arxiv.org/pdf/1612.00593.pdf)
