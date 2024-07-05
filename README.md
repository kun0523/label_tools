# label_tools
use to build datasets

## 数据集构建流程
1. 根据业务需要收集原始数据
2. 确定ROI
3. 裁剪ROI区域（可以训练一个简单模型去自动裁剪ROI）
4. 基于ROI图像标注目标对象
5. 构建数据飞轮优化模型

## 数据飞轮
1. 少批量标注数据（手动LabelMe标注）
2. 训练模型
3. 根据模型预测结果产生LabelMe Json
4. 在LabelMe中微调标注结果
4. 根据LabelMe Json 产生 Yolo 标注信息
5. 重新训练优化模型
