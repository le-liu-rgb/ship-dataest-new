# AIS Trajectory Prediction with DSTGTNet

## 项目概述：
本项目使用基于Transformer和图神经网络（GraphSAGE）的模型（DSTGTNet）来预测船舶轨迹。项目包含多个模块，分别负责数据加载、模型定义、训练和评估。

## 项目结构

### 主要文件及其功能

1. **GTais.py** - 主程序入口
   - 负责整个训练和评估流程
   - 循环遍历不同的时间值（5,10,15,20,25分钟）进行训练
   - 每次训练：
     - 重新加载配置
     - 设置学习率和保存路径
     - 加载数据
     - 初始化模型
     - 训练模型
     - 评估模型性能
     - 保存预测结果

2. **config_GTais.py** - 配置文件
   - 定义所有模型和训练的超参数
   - 包含数据集路径、模型结构参数、训练参数等
   - 可通过修改此文件调整实验设置

3. **STGT.py** - 模型定义
   - 实现STGTNet模型架构
   - 包含：
     - `CausalSelfAttention`：自注意力机制
     - `Block`：Transformer块
     - `GraphSAGE`：图神经网络模块（使用SAGEConv）
     - `STGTNet`：主模型，结合Transformer和GraphSAGE

4. **trainers.py** - 训练器
   - 实现`Trainer`类，负责模型训练和评估
   - 包含：
     - `sample()`：使用模型进行采样预测
     - `run_epoch()`：执行单个训练/验证周期
     - 损失记录和模型保存功能

5. **datasets.py** - 数据集处理
   - 定义`AISDataset`和`AISDataset_grad`类
   - 负责：
     - 加载AIS轨迹数据
     - 数据预处理
     - 生成适合模型输入的格式

6. **utils.py** - 工具函数
   - 提供辅助功能：
     - `set_seed()`：设置随机种子
     - `new_log()`：配置日志
     - `haversine()`：计算Haversine距离
     - `top_k_logits()`和`top_k_nearest_idx()`：采样相关函数

## 使用流程

### 1. 数据准备

1. 将AIS数据存储在`./data/ct_dma/`目录下
2. 数据应包含三个文件：
   - `ct_dma_train.pkl`：训练集
   - `ct_dma_valid.pkl`：验证集
   - `ct_dma_test.pkl`：测试集
3. 每个文件应包含船舶轨迹字典，格式为：
   ```python
   {
       "mmsi": 船舶ID,
       "traj": [
           [纬度, 经度, SOG, COG, 时间戳, MMSI],
           ...
       ]
   }
   ```

### 2. 配置设置 (config_GTais.py)

在运行前，根据需要调整以下配置：
- `retrain`: 是否重新训练模型 (True/False)
- `device`: 训练设备 (cuda:0/cpu)
- `time`: 预测时间间隔（主程序会覆盖此设置）
- `dataset_name`: 数据集名称
- `mode`: 预测模式 ("pos"/"pos_grad"/"mlp_pos"等)
- `sample_mode`: 采样模式 ("pos"/"pos_vicinity"/"velo")
- `n_head`和`n_layer`: Transformer头数和层数
- `learning_rate`: 学习率（主程序会根据时间值调整）

### 3. 运行主程序 (GTais.py)

```bash
python GTais.py
```
程序将：
1. 循环遍历时间值 [5, 10, 15, 20, 25]
2. 对于每个时间值：
   - 重新加载配置并更新相关参数
   - 设置学习率（根据时间值映射）
   - 创建唯一的保存目录
   - 加载并预处理数据
   - 初始化模型和训练器
   - 训练模型（如果retrain=True）
   - 评估模型性能
   - 保存预测误差结果
3. 生成预测误差曲线和CSV文件

### 4. 输出结果

- 模型检查点：保存在`./results/.../model.pt`
- 训练日志：保存在结果目录下的log文件
- 预测误差：
  - `prediction_error.png`：误差随时间变化曲线
  - `pred_errors.csv`：详细的误差数据
- 训练过程：
  - `training_loss.csv`：训练损失记录
  - `pred_mmsi_*.csv`：特定船舶的预测轨迹

## 关键功能说明

### GraphSAGE模块 (STGT.py)

- **功能**：建模船舶之间的时空关系
- **实现**：
  - `build_spatiotemporal_graph()`: 构建时空图结构
    - 考虑时间接近性（15分钟内）
    - 考虑空间接近性（1000米内）
  - `forward()`: 执行图卷积操作
- **用法**：在STGTNet模型中自动调用


### 采样函数 (trainers.py)

- **功能**：使用训练好的模型生成轨迹预测
- **参数**：
  - `temperature`: 控制采样随机性
  - `sample_mode`: 采样策略
  - `r_vicinity`: 邻域半径
  - `top_k`: 保留的top-k值
- **用法**：在评估阶段和训练过程中调用

### 训练循环 (trainers.py)

- **流程**：
  1. 准备数据加载器
  2. 前向传播计算损失
  3. 反向传播更新权重
  4. 定期保存模型和日志
  5. 每个epoch结束时进行采样和可视化

## 自定义和扩展

### 修改模型架构

1. 在`STGT.py`中修改`STGTNet`类
2. 调整图神经网络层数或类型
3. 修改Transformer结构（头数、层数等）

### 添加新数据集

1. 准备新数据集，格式与现有数据集一致
2. 在`config_GTais.py`中更新：
   - `dataset_name`
   - 相关区域参数（lat_min, lat_max等）
   - 数据路径

### 调整训练策略

   - 在`config_GTais.py`中修改：
   - `max_epochs`: 最大训练周期数
   - `batch_size`: 批次大小
   - `learning_rate`: 学习率
   - `lr_decay`: 是否使用学习率衰减


## 注意事项

1. 首次运行前确保安装所有依赖：
   ```bash
   pip install torch numpy matplotlib tqdm pickle logging torch_geometric
   ```
2. 使用GPU加速训练需配置`device = torch.device("cuda:0")`
3. 训练过程中会定期保存模型和可视化结果
4. 通过`TB_LOG`标志控制是否使用TensorBoard日志
