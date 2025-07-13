# -*- coding: utf-8 -*-
"""
HMM核心算法类（简化版）
从gotrackit项目中提取的隐马尔可夫模型关键算法
不依赖numpy，使用纯Python实现
"""

import math
from typing import Dict, List, Tuple, Optional, Union


class ViterbiSolverSimple:
    """
    Viterbi算法求解器（简化版）
    用于求解隐马尔可夫模型的最优状态序列
    """
    
    def __init__(self, observation_list: List[int], 
                 transition_matrices: Dict[int, List[List[float]]],
                 emission_matrices: Dict[int, List[float]], 
                 use_log_prob: bool = True):
        """
        初始化Viterbi求解器
        
        Args:
            observation_list: 观测序列
            transition_matrices: 状态转移概率矩阵字典
            emission_matrices: 观测概率矩阵字典
            use_log_prob: 是否使用对数概率
        """
        self.use_log_prob = use_log_prob
        self.observation_seq = observation_list
        self.T = len(observation_list)
        
        assert self.T > 1, '至少需要2个观测点'
        
        self.psi_matrices = {}  # 回溯路径
        self.zeta_matrices = {}  # 累积概率
        
        self.emission_matrices = emission_matrices
        self.transition_matrices = transition_matrices
        
    def initialize_model(self) -> None:
        """初始化模型"""
        # 默认均匀分布
        init_n = len(self.transition_matrices[self.observation_seq[0]])
        init_emission = self.emission_matrices[self.observation_seq[0]]
        
        if self.use_log_prob:
            self.zeta_matrices[self.observation_seq[0]] = [
                init_emission[i] - math.log(init_n) for i in range(len(init_emission))
            ]
        else:
            self.zeta_matrices[self.observation_seq[0]] = [
                (1.0 / init_n) * init_emission[i] for i in range(len(init_emission))
            ]
    
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
                self.zeta_matrices[obs],
                self.transition_matrices[obs],
                self.emission_matrices[next_obs]
            )
            
            # 找出每个状态的最优前驱状态
            optimal_prev_states = []
            next_probs = []
            
            for j in range(len(prob_matrix[0])):
                max_prob = float('-inf') if self.use_log_prob else 0.0
                best_prev_state = 0
                
                for k in range(len(prob_matrix)):
                    if self.use_log_prob:
                        if prob_matrix[k][j] > max_prob:
                            max_prob = prob_matrix[k][j]
                            best_prev_state = k
                    else:
                        if prob_matrix[k][j] > max_prob:
                            max_prob = prob_matrix[k][j]
                            best_prev_state = k
                
                optimal_prev_states.append(best_prev_state)
                next_probs.append(max_prob)
            
            # 记录回溯路径和最大概率
            self.psi_matrices[next_obs] = optimal_prev_states
            self.zeta_matrices[next_obs] = next_probs
        
        # 2. 后向回溯
        return self._backtrack()
    
    def _calculate_probability_matrix(self, zeta_current: List[float],
                                    transition_matrix: List[List[float]],
                                    emission_next: List[float]) -> List[List[float]]:
        """计算概率矩阵"""
        rows = len(transition_matrix)
        cols = len(transition_matrix[0])
        
        prob_matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                if self.use_log_prob:
                    prob_matrix[i][j] = (zeta_current[i] + 
                                       transition_matrix[i][j] + 
                                       emission_next[j])
                else:
                    prob_matrix[i][j] = (zeta_current[i] * 
                                       transition_matrix[i][j] * 
                                       emission_next[j])
        
        return prob_matrix
    
    def _backtrack(self) -> List[int]:
        """回溯求解最优路径"""
        state_sequence = []
        
        # 找出最后一个时刻的最优状态
        last_obs = self.observation_seq[-1]
        max_prob = float('-inf') if self.use_log_prob else 0.0
        last_state = 0
        
        for i, prob in enumerate(self.zeta_matrices[last_obs]):
            if self.use_log_prob:
                if prob > max_prob:
                    max_prob = prob
                    last_state = i
            else:
                if prob > max_prob:
                    max_prob = prob
                    last_state = i
        
        state_sequence.append(last_state)
        
        # 从后往前回溯
        current_state = last_state
        for i in range(self.T - 2, -1, -1):
            obs = self.observation_seq[i + 1]
            current_state = self.psi_matrices[obs][current_state]
            state_sequence.append(current_state)
        
        return state_sequence[::-1]


class HMMCoreSimple:
    """
    隐马尔可夫模型核心算法类（简化版）
    提取了关键的HMM算法，使用纯Python实现
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
        
    def calculate_transition_probability(self, distance_gap: float, 
                                       beta: Optional[float] = None,
                                       distance_param: Optional[float] = None) -> float:
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
            return math.exp(-distance_param * distance_gap / beta)
    
    def calculate_emission_probability(self, distance: float,
                                     sigma: Optional[float] = None,
                                     heading_factor: Optional[float] = None) -> float:
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
        heading_factor = heading_factor if heading_factor is not None else 1.0
        
        if self.use_log_prob:
            return (math.log(heading_factor) - 
                   0.5 * (self.distance_param * distance / sigma) ** 2)
        else:
            return (heading_factor * 
                   math.exp(-0.5 * (self.distance_param * distance / sigma) ** 2))
    
    def build_transition_matrix(self, from_states: List, to_states: List,
                               distance_gaps: List[float]) -> List[List[float]]:
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
        
        transition_matrix = [[0.0 for _ in range(n_to)] for _ in range(n_from)]
        
        for i in range(n_from):
            for j in range(n_to):
                idx = i * n_to + j
                if idx < len(distance_gaps):
                    transition_matrix[i][j] = self.calculate_transition_probability(
                        distance_gaps[idx]
                    )
        
        return transition_matrix
    
    def build_emission_matrix(self, states: List, distances: List[float],
                             heading_factors: Optional[List[float]] = None) -> List[float]:
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
        
        return emission_probs
    
    def fit(self, observation_sequence: List[int],
            state_candidates: Dict[int, List],
            distance_data: Dict,
            heading_data: Optional[Dict[int, List[float]]] = None) -> Tuple[bool, List[int]]:
        """
        训练HMM模型并求解最优状态序列
        
        Args:
            observation_sequence: 观测序列
            state_candidates: 每个观测点的候选状态
            distance_data: 距离数据
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
                    distances = [float(distances)]
                else:
                    distances = [float(d) for d in distances]
                
                heading_factors = heading_data.get(obs, [1.0] * len(states)) if heading_data else None
                
                emission_matrices[obs] = self.build_emission_matrix(
                    states, distances, heading_factors
                )
            
            # 3. 使用Viterbi算法求解
            self.solver = ViterbiSolverSimple(
                observation_sequence,
                transition_matrices,
                emission_matrices,
                self.use_log_prob
            )
            
            self.solver.initialize_model()
            optimal_states = self.solver.solve()
            
            # 存储矩阵用于后续预测
            self.transition_matrices = transition_matrices
            self.emission_matrices = emission_matrices
            
            return True, optimal_states
            
        except Exception as e:
            print(f"HMM训练过程中发生错误: {e}")
            return False, []
    
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
            emission_prob = self.emission_matrices[obs][state]
            
            if i > 0:
                # 转移概率
                prev_obs = observation_sequence[i-1]
                prev_state = state_sequence[i-1]
                transition_prob = self.transition_matrices[prev_obs][prev_state][state]
                
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
    print("HMM核心算法演示")
    print("="*50)
    
    # 创建HMM实例
    hmm = HMMCoreSimple(beta=30.0, sigma=20.0, use_log_prob=True)
    
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
    
    print("观测序列:", observation_sequence)
    print("状态候选:", state_candidates)
    print()
    
    # 训练模型
    success, optimal_states = hmm.fit(observation_sequence, state_candidates, distance_data)
    
    if success:
        print("✓ 模型训练成功")
        print(f"最优状态序列: {optimal_states}")
        
        # 计算序列概率
        prob = hmm.calculate_sequence_probability(observation_sequence, optimal_states)
        print(f"序列概率: {prob:.4f}")
        
        print()
        print("算法验证:")
        print("-" * 30)
        
        # 验证概率计算
        print("状态转移概率示例:")
        for i, distance_gap in enumerate([5.0, 8.0, 12.0]):
            trans_prob = hmm.calculate_transition_probability(distance_gap)
            print(f"  距离差={distance_gap:.1f} -> 概率={trans_prob:.4f}")
        
        print("\n观测概率示例:")
        for i, distance in enumerate([10.0, 15.0, 20.0]):
            emit_prob = hmm.calculate_emission_probability(distance)
            print(f"  观测距离={distance:.1f} -> 概率={emit_prob:.4f}")
        
        print("\n核心算法特点:")
        print("- 使用对数概率避免数值下溢")
        print("- Viterbi算法动态规划求解")
        print("- 基于距离的概率计算模型")
        print("- 支持方向因子调整")
        
    else:
        print("✗ 模型训练失败")


if __name__ == "__main__":
    create_hmm_example()