# -*- coding: utf-8 -*-
"""
HMM核心算法类
从gotrackit项目中提取的隐马尔可夫模型关键算法
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


class ViterbiSolver:
    """
    Viterbi算法求解器
    用于求解隐马尔可夫模型的最优状态序列
    """
    
    def __init__(self, observation_list: List[int], 
                 transition_matrices: Dict[int, np.ndarray],
                 emission_matrices: Dict[int, np.ndarray], 
                 use_log_prob: bool = True,
                 initial_prob: Optional[Dict[int, np.ndarray]] = None):
        """
        初始化Viterbi求解器
        
        Args:
            observation_list: 观测序列
            transition_matrices: 状态转移概率矩阵字典
            emission_matrices: 观测概率矩阵字典
            use_log_prob: 是否使用对数概率
            initial_prob: 初始概率分布
        """
        self.use_log_prob = use_log_prob
        self.observation_seq = observation_list
        self.T = len(observation_list)
        
        assert self.T > 1, '至少需要2个观测点'
        
        self.psi_matrices = {}  # 回溯路径
        self.zeta_matrices = {}  # 累积概率
        
        # 格式化观测概率矩阵
        self.emission_matrices = self._format_emission_matrices(emission_matrices)
        self.transition_matrices = self._format_transition_matrices(transition_matrices)
        self.initial_prob = initial_prob
        
    def _format_emission_matrices(self, emission_matrices: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """格式化观测概率矩阵"""
        formatted = {}
        for obs_seq, matrix in emission_matrices.items():
            if len(matrix.shape) == 1:
                formatted[obs_seq] = matrix.reshape(1, matrix.shape[0])
            else:
                formatted[obs_seq] = matrix
        return formatted
    
    def _format_transition_matrices(self, transition_matrices: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """格式化状态转移概率矩阵"""
        if len(transition_matrices) == 1:
            # 如果只有一个矩阵，则对所有时刻都使用同一个矩阵
            key = list(transition_matrices.keys())[0]
            return {k: transition_matrices[key] for k in self.observation_seq[:-1]}
        return transition_matrices
    
    def initialize_model(self) -> None:
        """初始化模型"""
        if self.initial_prob is None:
            # 默认均匀分布
            init_n = self.transition_matrices[self.observation_seq[0]].shape[0]
            init_emission = self.emission_matrices[self.observation_seq[0]]
            
            if self.use_log_prob:
                self.initial_prob = init_emission.astype(float) - np.log(init_n)
            else:
                self.initial_prob = (1 / init_n) * init_emission.astype(float)
        
        self.zeta_matrices[self.observation_seq[0]] = self.initial_prob
    
    def solve(self) -> List[int]:
        """
        使用Viterbi算法求解最优状态序列
        
        Returns:
            最优状态序列的索引列表
        """
        # 1. 前向传播
        for i, obs in enumerate(self.observation_seq[:-1]):
            next_obs = self.observation_seq[i + 1]
            
            # 计算下一时刻的概率
            prob_matrix = self._calculate_probability_matrix(
                self.zeta_matrices[obs].T,
                self.transition_matrices[obs],
                self.emission_matrices[next_obs]
            )
            
            # 找出每个状态的最优前驱状态
            optimal_prev_states = np.argmax(prob_matrix, axis=0)
            
            # 记录回溯路径和最大概率
            self.psi_matrices[next_obs] = optimal_prev_states
            self.zeta_matrices[next_obs] = np.array([
                prob_matrix[optimal_prev_states, range(len(optimal_prev_states))]
            ])
        
        # 2. 后向回溯
        return self._backtrack()
    
    def _calculate_probability_matrix(self, zeta_current: np.ndarray,
                                    transition_matrix: np.ndarray,
                                    emission_next: np.ndarray) -> np.ndarray:
        """计算概率矩阵"""
        if self.use_log_prob:
            return (zeta_current.astype(np.float32) + 
                   transition_matrix.astype(np.float32) + 
                   emission_next.astype(np.float32))
        else:
            return zeta_current * transition_matrix * emission_next
    
    def _backtrack(self) -> List[int]:
        """回溯求解最优路径"""
        state_sequence = []
        last_max_state = -1
        
        for i in range(self.T - 1, 0, -1):
            obs = self.observation_seq[i]
            
            if i == self.T - 1:
                # 回溯起点
                start_max_state = np.argmax(self.zeta_matrices[obs])
                state_sequence.append(start_max_state)
                last_max_state = self.psi_matrices[obs][start_max_state]
                state_sequence.append(last_max_state)
            else:
                start_max_state = last_max_state
                last_max_state = self.psi_matrices[obs][start_max_state]
                state_sequence.append(last_max_state)
        
        return state_sequence[::-1]


class HMMCore:
    """
    隐马尔可夫模型核心算法类
    提取了关键的HMM算法，可用于各种序列标注和状态估计任务
    """
    
    def __init__(self, beta: float = 30.1, sigma: float = 20.0, 
                 distance_param: float = 0.1, use_log_prob: bool = True):
        """
        初始化HMM核心类
        
        Args:
            beta: 状态转移概率的beta参数
            sigma: 观测概率的sigma参数
            distance_param: 距离参数
            use_log_prob: 是否使用对数概率
        """
        self.beta = beta
        self.sigma = sigma
        self.distance_param = distance_param
        self.use_log_prob = use_log_prob
        
        # 存储计算结果
        self.transition_matrices = {}
        self.emission_matrices = {}
        self.solver = None
        
    def calculate_transition_probability(self, distance_gap: Union[float, np.ndarray], 
                                       beta: Optional[float] = None,
                                       distance_param: Optional[float] = None) -> Union[float, np.ndarray]:
        """
        计算状态转移概率
        
        Args:
            distance_gap: 距离差值 (straight_distance - route_distance)
            beta: beta参数
            distance_param: 距离参数
            
        Returns:
            状态转移概率
        """
        beta = beta or self.beta
        distance_param = distance_param or self.distance_param
        
        if self.use_log_prob:
            return -distance_param * distance_gap / beta
        else:
            return np.exp(-distance_param * distance_gap / beta)
    
    def calculate_emission_probability(self, distance: Union[float, np.ndarray],
                                     sigma: Optional[float] = None,
                                     heading_factor: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
        """
        计算观测概率
        
        Args:
            distance: 观测距离
            sigma: sigma参数
            heading_factor: 方向因子
            
        Returns:
            观测概率
        """
        sigma = sigma or self.sigma
        heading_factor = heading_factor if heading_factor is not None else np.ones_like(distance)
        
        if self.use_log_prob:
            return (np.log(heading_factor) - 
                   0.5 * (self.distance_param * distance / sigma) ** 2)
        else:
            return (heading_factor * 
                   np.exp(-0.5 * (self.distance_param * distance / sigma) ** 2))
    
    def build_transition_matrix(self, from_states: List, to_states: List,
                               distance_gaps: List[float]) -> np.ndarray:
        """
        构建状态转移概率矩阵
        
        Args:
            from_states: 起始状态列表
            to_states: 目标状态列表
            distance_gaps: 距离差值列表
            
        Returns:
            状态转移概率矩阵
        """
        n_from = len(from_states)
        n_to = len(to_states)
        
        transition_matrix = np.zeros((n_from, n_to))
        
        for i, from_state in enumerate(from_states):
            for j, to_state in enumerate(to_states):
                idx = i * n_to + j
                if idx < len(distance_gaps):
                    transition_matrix[i, j] = self.calculate_transition_probability(
                        distance_gaps[idx]
                    )
        
        return transition_matrix
    
    def build_emission_matrix(self, states: List, distances: List[float],
                             heading_factors: Optional[List[float]] = None) -> np.ndarray:
        """
        构建观测概率矩阵
        
        Args:
            states: 状态列表
            distances: 距离列表
            heading_factors: 方向因子列表
            
        Returns:
            观测概率矩阵
        """
        if heading_factors is None:
            heading_factors = [1.0] * len(distances)
        
        emission_probs = []
        for i, state in enumerate(states):
            if i < len(distances):
                prob = self.calculate_emission_probability(
                    distances[i], heading_factor=heading_factors[i]
                )
                emission_probs.append(prob)
        
        return np.array(emission_probs).reshape(1, -1)
    
    def fit(self, observation_sequence: List[int],
            state_candidates: Dict[int, List],
            distance_data: Dict[str, List[float]],
            heading_data: Optional[Dict[int, List[float]]] = None) -> Tuple[bool, List[int]]:
        """
        训练HMM模型并求解最优状态序列
        
        Args:
            observation_sequence: 观测序列
            state_candidates: 每个观测点的候选状态
            distance_data: 距离数据，包含'emission_distances'和'transition_distances'
            heading_data: 方向数据
            
        Returns:
            (是否成功, 最优状态序列)
        """
        try:
            # 1. 构建转移概率矩阵
            transition_matrices = {}
            for i, obs in enumerate(observation_sequence[:-1]):
                next_obs = observation_sequence[i + 1]
                from_states = state_candidates[obs]
                to_states = state_candidates[next_obs]
                
                # 获取对应的距离数据
                start_idx = i * len(from_states) * len(to_states)
                end_idx = start_idx + len(from_states) * len(to_states)
                distance_gaps = distance_data['transition_distances'][start_idx:end_idx]
                
                transition_matrices[obs] = self.build_transition_matrix(
                    from_states, to_states, distance_gaps
                )
            
            # 2. 构建观测概率矩阵
            emission_matrices = {}
            for obs in observation_sequence:
                states = state_candidates[obs]
                distances = distance_data['emission_distances'][obs]
                if not isinstance(distances, list):
                    distances = [float(distances)]  # 确保distances是float列表
                else:
                    distances = [float(d) for d in distances]  # 确保所有元素都是float
                heading_factors = heading_data.get(obs, [1.0] * len(states)) if heading_data else None
                
                emission_matrices[obs] = self.build_emission_matrix(
                    states, distances, heading_factors
                )
            
            # 3. 使用Viterbi算法求解
            self.solver = ViterbiSolver(
                observation_sequence,
                transition_matrices,
                emission_matrices,
                self.use_log_prob
            )
            
            self.solver.initialize_model()
            optimal_states = self.solver.solve()
            
            return True, optimal_states
            
        except Exception as e:
            print(f"HMM训练过程中发生错误: {e}")
            return False, []
    
    def predict(self, observation_sequence: List[int]) -> List[int]:
        """
        使用已训练的模型进行预测
        
        Args:
            observation_sequence: 观测序列
            
        Returns:
            预测的状态序列
        """
        if self.solver is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 使用现有的转移和观测矩阵进行预测
        solver = ViterbiSolver(
            observation_sequence,
            self.transition_matrices,
            self.emission_matrices,
            self.use_log_prob
        )
        
        solver.initialize_model()
        return solver.solve()
    
    def get_probability_matrices(self) -> Dict[str, Dict]:
        """
        获取概率矩阵
        
        Returns:
            包含转移概率矩阵和观测概率矩阵的字典
        """
        return {
            'transition_matrices': self.transition_matrices,
            'emission_matrices': self.emission_matrices
        }
    
    def calculate_sequence_probability(self, observation_sequence: List[int],
                                     state_sequence: List[int]) -> float:
        """
        计算给定观测序列和状态序列的概率
        
        Args:
            observation_sequence: 观测序列
            state_sequence: 状态序列
            
        Returns:
            序列概率
        """
        if len(observation_sequence) != len(state_sequence):
            raise ValueError("观测序列和状态序列长度不匹配")
        
        if self.solver is None:
            raise ValueError("模型尚未训练")
        
        total_prob = 0.0 if self.use_log_prob else 1.0
        
        for i, obs in enumerate(observation_sequence):
            state = state_sequence[i]
            
            # 观测概率
            emission_prob = self.emission_matrices[obs][0, state]
            
            if i > 0:
                # 转移概率
                prev_obs = observation_sequence[i-1]
                prev_state = state_sequence[i-1]
                transition_prob = self.transition_matrices[prev_obs][prev_state, state]
                
                if self.use_log_prob:
                    total_prob += transition_prob + emission_prob
                else:
                    total_prob *= transition_prob * emission_prob
            else:
                if self.use_log_prob:
                    total_prob += emission_prob
                else:
                    total_prob *= emission_prob
        
        return total_prob


def create_hmm_example():
    """
    创建一个简单的HMM使用示例
    """
    # 创建HMM实例
    hmm = HMMCore(beta=30.0, sigma=20.0, use_log_prob=True)
    
    # 示例数据
    observation_sequence = [0, 1, 2, 3]
    state_candidates = {
        0: [0, 1, 2],
        1: [0, 1, 2],
        2: [0, 1],
        3: [0, 1, 2]
    }
    
    # 距离数据
    distance_data = {
        'emission_distances': {
            0: [10.0, 15.0, 20.0],
            1: [12.0, 8.0, 25.0],
            2: [5.0, 18.0],
            3: [15.0, 10.0, 22.0]
        },
        'transition_distances': [
            # 从观测点0到观测点1的转移距离
            5.0, 8.0, 12.0, 7.0, 3.0, 15.0, 10.0, 6.0, 20.0,
            # 从观测点1到观测点2的转移距离
            4.0, 9.0, 6.0, 11.0,
            # 从观测点2到观测点3的转移距离
            8.0, 12.0, 5.0, 14.0, 7.0, 16.0
        ]
    }
    
    # 训练模型
    success, optimal_states = hmm.fit(observation_sequence, state_candidates, distance_data)
    
    if success:
        print(f"最优状态序列: {optimal_states}")
        
        # 计算序列概率
        prob = hmm.calculate_sequence_probability(observation_sequence, optimal_states)
        print(f"序列概率: {prob}")
        
        # 获取概率矩阵
        matrices = hmm.get_probability_matrices()
        print(f"转移矩阵数量: {len(matrices['transition_matrices'])}")
        print(f"观测矩阵数量: {len(matrices['emission_matrices'])}")
    else:
        print("模型训练失败")


if __name__ == "__main__":
    create_hmm_example()