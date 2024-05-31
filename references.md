Model based:
- https://github.com/NVlabs/FoundationPose/issues/110 unit of depth images
- https://github.com/NVlabs/FoundationPose/issues/143 realsense coordinate system
- https://github.com/NVlabs/FoundationPose/issues/44 详细设置说明
- https://github.com/NVlabs/FoundationPose/issues/140 堆叠场景

Model free：
- https://github.com/NVlabs/FoundationPose/issues/109 重建模型
- https://github.com/NVlabs/FoundationPose/issues/142 重建模型

先用一个简单的模型试一下吧， 确保程序无误再测试我的真实物体. 
- 检查debug的输出是否正确, 比如看这个:https://github.com/NVlabs/FoundationPose/issues/132.
- vis_refiner.png可以显示每一次的姿态估计结果和对应的模型可视化结果
- scene_complete以及scene_raw等点云文件可以通过cloud compare打开, 是带纹理的, 可以学习是怎么生成的.