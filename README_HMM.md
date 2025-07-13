# HMM核心算法类

## 概述

本项目从gotrackit仓库中提取了隐马尔可夫模型（HMM）的核心算法，封装成了一个独立的类`HMMCore`。该类去除了GPS轨迹匹配的特定实现，使其能够应用于各种序列标注和状态估计任务。

## 主要特性

### 1. 核心算法提取

从原始项目中提取了以下关键算法：

- **Viterbi算法**：用于求解最优状态序列
- **状态转移概率计算**：基于距离差值的概率计算
- **观测概率计算**：基于观测距离和方向因子的概率计算
- **概率矩阵构建**：动态构建转移和观测概率矩阵

### 2. 类结构

#### ViterbiSolver类
- 独立的Viterbi算法实现
- 支持对数概率计算，避免数值下溢
- 前向传播和后向回溯求解最优路径

#### HMMCore类
- 主要的HMM模型接口
- 集成了概率计算、矩阵构建和求解功能
- 支持模型训练和预测

## 使用方法

### 基本使用

```python
from hmm_core import HMMCore

# 创建HMM实例
hmm = HMMCore(beta=30.0, sigma=20.0, use_log_prob=True)

# 准备数据
observation_sequence = [0, 1, 2, 3]
state_candidates = {
    0: [0, 1, 2],
    1: [0, 1, 2], 
    2: [0, 1],
    3: [0, 1, 2]
}

distance_data = {
    'emission_distances': {
        0: [10.0, 15.0, 20.0],
        1: [12.0, 8.0, 25.0],
        2: [5.0, 18.0],
        3: [15.0, 10.0, 22.0]
    },
    'transition_distances': [
        # 状态转移距离数据
        5.0, 8.0, 12.0, 7.0, 3.0, 15.0, 10.0, 6.0, 20.0,
        4.0, 9.0, 6.0, 11.0,
        8.0, 12.0, 5.0, 14.0, 7.0, 16.0
    ]
}

# 训练模型并求解
success, optimal_states = hmm.fit(observation_sequence, state_candidates, distance_data)

if success:
    print(f"最优状态序列: {optimal_states}")
    
    # 计算序列概率
    prob = hmm.calculate_sequence_probability(observation_sequence, optimal_states)
    print(f"序列概率: {prob}")
```

### 高级功能

#### 1. 添加方向信息
```python
# 添加方向因子
heading_data = {
    0: [1.0, 0.8, 0.6],
    1: [0.9, 1.0, 0.7],
    2: [1.0, 0.8],
    3: [0.8, 1.0, 0.9]
}

success, optimal_states = hmm.fit(
    observation_sequence, 
    state_candidates, 
    distance_data,
    heading_data=heading_data
)
```

#### 2. 获取概率矩阵
```python
matrices = hmm.get_probability_matrices()
transition_matrices = matrices['transition_matrices']
emission_matrices = matrices['emission_matrices']
```

#### 3. 使用训练好的模型进行预测
```python
# 使用已训练的模型进行新序列的预测
new_observation_sequence = [0, 1, 2]
predicted_states = hmm.predict(new_observation_sequence)
```

## 核心算法说明

### 1. 状态转移概率

```python
def calculate_transition_probability(self, distance_gap, beta=None, distance_param=None):
    """
    计算状态转移概率
    
    Args:
        distance_gap: 距离差值 (直线距离 - 路径距离)
        beta: 控制概率衰减的参数
        distance_param: 距离权重参数
    
    Returns:
        状态转移概率
    """
```

状态转移概率基于距离差值计算：
- 距离差值越小，转移概率越高
- 支持对数概率计算：`log_prob = -distance_param * distance_gap / beta`

### 2. 观测概率

```python
def calculate_emission_probability(self, distance, sigma=None, heading_factor=None):
    """
    计算观测概率
    
    Args:
        distance: 观测距离
        sigma: 标准差参数
        heading_factor: 方向因子
    
    Returns:
        观测概率
    """
```

观测概率基于观测距离和方向因子计算：
- 距离越近，观测概率越高
- 方向因子用于调整方向匹配度的影响

### 3. Viterbi算法

Viterbi算法用于求解最优状态序列：
1. **前向传播**：计算每个时刻每个状态的最大概率
2. **后向回溯**：从最后一个时刻开始，回溯最优路径

## 参数说明

### HMMCore参数
- `beta`: 状态转移概率的衰减参数，默认30.1
- `sigma`: 观测概率的标准差参数，默认20.0  
- `distance_param`: 距离权重参数，默认0.1
- `use_log_prob`: 是否使用对数概率，默认True

### 数据格式

#### observation_sequence
观测序列，为整数列表，如：`[0, 1, 2, 3]`

#### state_candidates
每个观测点的候选状态字典：
```python
{
    0: [0, 1, 2],  # 观测点0的候选状态
    1: [0, 1, 2],  # 观测点1的候选状态
    2: [0, 1],     # 观测点2的候选状态
    3: [0, 1, 2]   # 观测点3的候选状态
}
```

#### distance_data
距离数据字典：
```python
{
    'emission_distances': {
        0: [10.0, 15.0, 20.0],  # 观测点0到各候选状态的距离
        1: [12.0, 8.0, 25.0],   # 观测点1到各候选状态的距离
        # ...
    },
    'transition_distances': [
        # 状态转移的距离数据，按顺序排列
        5.0, 8.0, 12.0, 7.0, 3.0, 15.0, 10.0, 6.0, 20.0,
        # ...
    ]
}
```

## 应用场景

这个HMM核心类可以应用于多种场景：

1. **序列标注**：如词性标注、命名实体识别
2. **状态估计**：如设备状态监控、用户行为分析
3. **轨迹分析**：如路径规划、运动轨迹分析
4. **信号处理**：如语音识别、手写识别

## 性能优化

1. **对数概率**：避免数值下溢问题
2. **矩阵化计算**：使用NumPy进行高效的矩阵运算
3. **内存管理**：合理的数据结构设计，减少内存占用

## 依赖项

- NumPy: 用于数值计算和矩阵操作
- typing: 提供类型提示支持

## 注意事项

1. 确保观测序列和状态候选的数据格式正确
2. 距离数据的长度要与状态转移次数匹配
3. 参数设置要根据具体应用场景调整
4. 对于大规模数据，考虑使用对数概率以避免数值问题

## 示例运行

运行示例代码：
```bash
python hmm_core.py
```

这将运行内置的示例，展示HMM核心类的基本用法。