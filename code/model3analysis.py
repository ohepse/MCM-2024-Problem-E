import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

# 设置绘图风格
plt.style.use('ggplot')

# ==========================================
# 1. 模型基础参数 (来自 model3.py)
# ==========================================
# 经济与成本参数
BASE_ECONOMIC_VALUE = 171e6  
GROWTH_RATE = 0.02           
DISCOUNT_RATE = 0.04         
YEARS = np.arange(2025, 2051)
COST_INITIAL_MEASURE = 550e6 
COST_MAINTENANCE_YEARLY = 3e6 
AVG_LOSS_PER_EVENT_UNPROTECTED = 5e5 
RISK_REDUCTION_RATE = 0.90 
ALPHA_FACTOR_BASELINE = 3.0e6  # 基准影子价格

# 气候风险模型函数
def get_weather_frequency(year_offset_from_1990):
    return 5.87 * np.exp(0.0122 * year_offset_from_1990)

# ==========================================
# 2. 核心计算逻辑封装
# ==========================================

def calculate_base_metrics():
    """计算不随权重和Alpha变化的固定经济与成本指标"""
    cumulative_econ_npv = 0
    cumulative_loss_protected_npv = 0
    cumulative_maintenance_npv = 0
    
    for i, year in enumerate(YEARS):
        # A. 经济价值 (V_econ)
        annual_revenue = BASE_ECONOMIC_VALUE * ((1 + GROWTH_RATE) ** i)
        discount_factor = 1 / ((1 + DISCOUNT_RATE) ** i)
        cumulative_econ_npv += annual_revenue * discount_factor
        
        # B. 风险计算 (Risk)
        t_from_1990 = 35 + i # 2025年对应 t=35
        lambda_freq = get_weather_frequency(t_from_1990)
        
        eal_unprotected = lambda_freq * AVG_LOSS_PER_EVENT_UNPROTECTED
        eal_protected = eal_unprotected * (1 - RISK_REDUCTION_RATE)
        cumulative_loss_protected_npv += eal_protected * discount_factor
        
        # C. 维护成本 (Maintenance)
        cumulative_maintenance_npv += COST_MAINTENANCE_YEARLY * discount_factor
        
    V_econ = cumulative_econ_npv
    TC = COST_INITIAL_MEASURE + cumulative_maintenance_npv + cumulative_loss_protected_npv
    
    return V_econ, TC

def calculate_cultural_score(weights):
    """根据权重计算 The Alamo 的文化得分"""
    # 固定输入数据
    data_culture = {
        'Landmark': ['The Alamo', 'North Star Mall', 'City Park'],
        'Age_Years': [300, 40, 20],           
        'UNESCO_Status': [10, 0, 0],          
        'Visitor_Millions': [1.6, 10.0, 0.5], 
        'Symbolism': [10, 2, 3]               
    }
    df_culture = pd.DataFrame(data_culture)
    numeric_cols = df_culture.columns[1:]
    
    # 归一化
    df_norm = df_culture[numeric_cols].copy()
    for col in numeric_cols:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val - min_val != 0:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 1.0
            
    # 加权得分
    df_culture['Score'] = (df_norm * weights).sum(axis=1) * 100
    target_score = df_culture.loc[df_culture['Landmark'] == 'The Alamo', 'Score'].values[0]
    return target_score

def get_baseline_weights():
    """计算基准熵权"""
    data_culture = {
        'Age_Years': [300, 40, 20],           
        'UNESCO_Status': [10, 0, 0],          
        'Visitor_Millions': [1.6, 10.0, 0.5], 
        'Symbolism': [10, 2, 3]               
    }
    df = pd.DataFrame(data_culture)
    
    # 归一化
    df_norm = df.copy()
    for col in df.columns:
        df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
    # 熵权计算
    P = df_norm + 1e-10
    P = P.div(P.sum(axis=0), axis=1)
    k = 1.0 / np.log(len(df))
    E = -k * (P * np.log(P)).sum(axis=0)
    D = 1 - E
    weights = D / D.sum()
    return weights

# ==========================================
# 3. 执行分析
# ==========================================

# 预计算固定值
V_econ_base, TC_base = calculate_base_metrics()
weights_base = get_baseline_weights()
score_base = calculate_cultural_score(weights_base)

# --- 分析 1: Alpha (影子价格) 敏感性分析 ---
# 扫描 Alpha 从 0 到 1000万
alpha_values = np.linspace(0, 10e6, 100) 
pi_values_alpha = []

for alpha in alpha_values:
    tv = V_econ_base + score_base * alpha
    pi = tv / TC_base
    pi_values_alpha.append(pi)

# 计算盈亏平衡点 (PI = 1)
# Alpha = (TC - V_econ) / Score
alpha_breakeven = (TC_base - V_econ_base) / score_base

# --- 分析 2: 权重鲁棒性分析 (Monte Carlo) ---
np.random.seed(42)
n_simulations = 1000
pi_values_robustness = []
perturbation_level = 0.50 # +/- 20% 的扰动

for _ in range(n_simulations):
    # 对权重施加随机扰动
    perturbations = np.random.uniform(1 - perturbation_level, 1 + perturbation_level, len(weights_base))
    weights_new = weights_base * perturbations
    # 重新归一化
    weights_new = weights_new / weights_new.sum()
    
    # 计算新的 PI (保持 Alpha 为基准值)
    score_new = calculate_cultural_score(weights_new)
    tv_new = V_econ_base + score_new * ALPHA_FACTOR_BASELINE
    pi_new = tv_new / TC_base
    pi_values_robustness.append(pi_new)

# ==========================================
# 4. 绘图
# ==========================================
fig, ax2 = plt.subplots(1, 1, figsize=(6, 6))

# 图 2: 权重鲁棒性分布
ax2.hist(pi_values_robustness, bins=30, color='#2ecc71', alpha=0.7, edgecolor="#2ecc71")
ax2.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='Threshold (PI=1.0)')
ax2.set_title(f'Robustness Analysis: Weight Perturbation (± {perturbation_level*100}%)')
ax2.set_xlabel('Preservation Index (PI)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()