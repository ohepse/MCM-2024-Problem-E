import numpy as np
import pandas as pd
from scipy.stats import entropy

# ==========================================
# 1. 配置参数与真实数据录入 (Data Entry)
# ==========================================

# --- A. 经济数据 (来源: NPS Report 2024 & San Antonio Forecast) ---
# 圣安东尼奥布道所(Missions)的年度经济贡献: ~$171 Million (2024)
BASE_ECONOMIC_VALUE = 171e6  
# 旅游业年增长率预测: ~2.0% (基于市政府财政预测 1.6%-2.4%)
GROWTH_RATE = 0.02           
# 贴现率 (Social Discount Rate)
DISCOUNT_RATE = 0.04         
# 评估年限 (2025-2050)
YEARS = np.arange(2025, 2051)
t_index = np.arange(len(YEARS)) # t=0, 1, 2...

# --- B. 保护成本数据 (来源: The Alamo Plan) ---
# 阿拉莫计划总预算: $550 Million (最新估值)
COST_INITIAL_MEASURE = 550e6 
# 年维护成本 (假设为造价的 0.5% - 1%)
COST_MAINTENANCE_YEARLY = 3e6 

# --- C. 天气风险模型 (User Provided) ---
# 您的拟合公式: lambda = 5.87 * e^(0.0122 * t)
# 这里 t 是年份索引 (0=1990?), 我们需要调整 t 以匹配未来预测
# 假设您的公式 t=0 对应 1990年。
# 2025年对应 t = 35
def get_weather_frequency(year_offset_from_1990):
    return 5.87 * np.exp(0.0122 * year_offset_from_1990)

# --- D. 损失模型假设 (Loss Assumptions) ---
# 假设每次"极端天气事件"造成的平均未保护损失 (Unprotected Loss)
# 这是一个估算值：大部分事件损失小，偶发大洪水。
AVG_LOSS_PER_EVENT_UNPROTECTED = 5e5  # $500k per event average
# 实施保护措施(The Alamo Plan + 防洪渠)后的损失减免率
RISK_REDUCTION_RATE = 0.90  # 90% reduction

# --- E. 文化价值参数 ---
# 支付意愿系数 (Shadow Price of Culture)
# 假设每1分文化得分 = 社会愿意支付 300万美元 (敏感性参数)
ALPHA_FACTOR = 3.0e6 

# ==========================================
# 2. 核心计算函数 (Core Functions)
# ==========================================

def calculate_entropy_weights(df_matrix):
    """
    使用熵权法计算各指标权重
    """
    # 1. 归一化 (Min-Max Normalization)
    df_norm = df_matrix.copy()
    for col in df_matrix.columns:
        if df_matrix[col].dtype != 'object':
            min_val = df_matrix[col].min()
            max_val = df_matrix[col].max()
            # 避免除以0
            if max_val - min_val == 0:
                df_norm[col] = 1.0
            else:
                df_norm[col] = (df_matrix[col] - min_val) / (max_val - min_val)
    
    # 2. 计算比重 P_ij
    # 加上微小值避免 log(0)
    P = df_norm.select_dtypes(include=np.number) + 1e-10
    P = P.div(P.sum(axis=0), axis=1)
    
    # 3. 计算熵值 e_j
    k = 1.0 / np.log(len(df_matrix))
    E = -k * (P * np.log(P)).sum(axis=0)
    
    # 4. 计算差异系数 d_j
    D = 1 - E
    
    # 5. 计算权重 w_j
    weights = D / D.sum()
    return weights, df_norm

def run_preservation_model():
    # -------------------------------
    # Step 1: 建立文化评分卡 (AHP/Entropy Input)
    # -------------------------------
    # 选取对比对象：阿拉莫 vs 当地商场 vs 普通公园
    data_culture = {
        'Landmark': ['The Alamo', 'North Star Mall', 'City Park'],
        'Age_Years': [300, 40, 20],           # 历史越久越好
        'UNESCO_Status': [10, 0, 0],          # 世遗=10
        'Visitor_Millions': [1.6, 10.0, 0.5], # 游客量 (商场其实人很多，这是反直觉的干扰项)
        'Symbolism': [10, 2, 3]               # 象征意义 (1-10)
    }
    df_culture = pd.DataFrame(data_culture)
    
    # 计算权重
    numeric_cols = df_culture.columns[1:]
    weights, df_norm = calculate_entropy_weights(df_culture[numeric_cols])
    
    # 计算综合得分 (0-100分制)
    df_culture['Score'] = (df_norm[numeric_cols] * weights).sum(axis=1) * 100
    
    target_score = df_culture.loc[df_culture['Landmark'] == 'The Alamo', 'Score'].values[0]
    
    print("--- 1. 文化价值评估 (Cultural Model) ---")
    print("指标权重:\n", weights)
    print(f"\nThe Alamo 文化得分: {target_score:.2f} / 100")
    print(f"货币化文化价值 (V_culture * alpha): ${target_score * ALPHA_FACTOR / 1e6:.2f} Million")

    # -------------------------------
    # Step 2: 经济与风险的时间序列模拟
    # -------------------------------
    simulation_data = []
    
    cumulative_econ_npv = 0
    cumulative_loss_unprotected_npv = 0
    cumulative_loss_protected_npv = 0
    cumulative_maintenance_npv = 0
    
    for i, year in enumerate(YEARS):
        # A. 经济价值 (V_econ)
        annual_revenue = BASE_ECONOMIC_VALUE * ((1 + GROWTH_RATE) ** i)
        discount_factor = 1 / ((1 + DISCOUNT_RATE) ** i)
        pv_revenue = annual_revenue * discount_factor
        cumulative_econ_npv += pv_revenue
        
        # B. 风险计算 (E[L])
        # 使用您的公式：t 从 1990年开始算起。2025年是 t=35
        t_from_1990 = 35 + i
        lambda_freq = get_weather_frequency(t_from_1990)
        
        # 预期年度损失 = 频率 * 单次平均损失
        eal_unprotected = lambda_freq * AVG_LOSS_PER_EVENT_UNPROTECTED
        eal_protected = eal_unprotected * (1 - RISK_REDUCTION_RATE)
        
        pv_loss_unprotected = eal_unprotected * discount_factor
        cumulative_loss_unprotected_npv += pv_loss_unprotected
        pv_loss_protected = eal_protected * discount_factor
        cumulative_loss_protected_npv += pv_loss_protected
        
        # C. 维护成本
        pv_maintenance = COST_MAINTENANCE_YEARLY * discount_factor
        cumulative_maintenance_npv += pv_maintenance
        
        simulation_data.append({
            'Year': year,
            'Lambda(Freq)': lambda_freq,
            'EAL_Unprotected': eal_unprotected,
            'EAL_Protected': eal_protected
        })
        
    df_sim = pd.DataFrame(simulation_data)
    
    # -------------------------------
    # Step 3: 计算保护指数 (PI)
    # -------------------------------
    
    # 总收益 (Total Value) = 经济NPV + 文化货币化价值
    V_econ = cumulative_econ_npv
    V_culture_monetized = target_score * ALPHA_FACTOR
    TV = V_econ + V_culture_monetized
    
    # 总成本 (Total Cost) = 初始投资 + 维护NPV + 剩余风险NPV
    # 注意：这里的成本是指“为了保护它而付出的代价”
    TC = COST_INITIAL_MEASURE + cumulative_maintenance_npv + cumulative_loss_protected_npv
    
    # 资金缺口 (Funding Gap) calculation
    # 商业视角：利润 = V_econ - (Initial + Maintenance + Residual Risk)
    commercial_profit = V_econ - TC
    
    PI = TV / TC
    
    print("\n--- 2. 成本效益分析 (Cost-Benefit Analysis) ---")
    print(f"评估周期: {len(YEARS)} 年 (2025-{YEARS[-1]})")
    print(f"极端天气频率 (2025): {df_sim.iloc[0]['Lambda(Freq)']:.2f} 次/年")
    print(f"极端天气频率 (2050): {df_sim.iloc[-1]['Lambda(Freq)']:.2f} 次/年")
    print("-" * 30)
    print(f"总经济价值 (V_econ): ${V_econ/1e6:.2f} M")
    print(f"总文化价值 (V_culture): ${V_culture_monetized/1e6:.2f} M (Alpha=${ALPHA_FACTOR/1e6:.1f}M/pt)")
    print(f"总社会价值 (TV): ${TV/1e6:.2f} M")
    print("-" * 30)
    print(f"实施成本 (C_measure): ${COST_INITIAL_MEASURE/1e6:.2f} M")
    print(f"维护成本 (NPV): ${cumulative_maintenance_npv/1e6:.2f} M")
    print(f"剩余风险 (Residual Risk): ${cumulative_loss_protected_npv/1e6:.2f} M")
    print(f"总项目成本 (TC): ${TC/1e6:.2f} M")
    print("-" * 30)
    print(f"商业盈亏 (Commercial Net): ${commercial_profit/1e6:.2f} M")
    print(f"保护指数 (PI): {PI:.2f}")

    if PI > 1:
        print("\n[结论]: 建议保护 (PI > 1)")
        if commercial_profit < 0:
            print(f"[政策建议]: 存在商业亏损。建议申请政府补贴 ${abs(commercial_profit)/1e6:.2f} M")
    else:
        print("\n[结论]: 建议放弃 (PI < 1)")

    return df_sim

# 执行模型
df_result = run_preservation_model()