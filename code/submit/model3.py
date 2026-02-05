import numpy as np
import pandas as pd
from scipy.stats import entropy

BASE_ECONOMIC_VALUE = 171e6  
GROWTH_RATE = 0.02           
DISCOUNT_RATE = 0.04         
YEARS = np.arange(2025, 2051)
t_index = np.arange(len(YEARS))
COST_INITIAL_MEASURE = 550e6 
COST_MAINTENANCE_YEARLY = 3e6 
def get_weather_frequency(year_offset_from_1990):
    return 5.87 * np.exp(0.0122 * year_offset_from_1990)
AVG_LOSS_PER_EVENT_UNPROTECTED = 5e5 
RISK_REDUCTION_RATE = 0.90  
ALPHA_FACTOR = 3.0e6 
def calculate_entropy_weights(df_matrix):
    df_norm = df_matrix.copy()
    for col in df_matrix.columns:
        if df_matrix[col].dtype != 'object':
            min_val = df_matrix[col].min()
            max_val = df_matrix[col].max()
            if max_val - min_val == 0:
                df_norm[col] = 1.0
            else:
                df_norm[col] = (df_matrix[col] - min_val) / (max_val - min_val)
    P = df_norm.select_dtypes(include=np.number) + 1e-10
    P = P.div(P.sum(axis=0), axis=1)
    k = 1.0 / np.log(len(df_matrix))
    E = -k * (P * np.log(P)).sum(axis=0)
    D = 1 - E
    weights = D / D.sum()
    return weights, df_norm
def run_preservation_model():
    data_culture = {
        'Landmark': ['The Alamo', 'North Star Mall', 'City Park'],
        'Age_Years': [300, 40, 20],           # 历史越久越好
        'UNESCO_Status': [10, 0, 0],          # 世遗=10
        'Visitor_Millions': [1.6, 10.0, 0.5], # 游客量 (商场其实人很多，这是反直觉的干扰项)
        'Symbolism': [10, 2, 3]               # 象征意义 (1-10)
    }
    df_culture = pd.DataFrame(data_culture)
    numeric_cols = df_culture.columns[1:]
    weights, df_norm = calculate_entropy_weights(df_culture[numeric_cols])
    df_culture['Score'] = (df_norm[numeric_cols] * weights).sum(axis=1) * 100
    target_score = df_culture.loc[df_culture['Landmark'] == 'The Alamo', 'Score'].values[0]
    print("--- 1. 文化价值评估 (Cultural Model) ---")
    print("指标权重:\n", weights)
    print(f"\nThe Alamo 文化得分: {target_score:.2f} / 100")
    print(f"货币化文化价值 (V_culture * alpha): ${target_score * ALPHA_FACTOR / 1e6:.2f} Million")

    simulation_data = []
    cumulative_econ_npv = 0
    cumulative_loss_unprotected_npv = 0
    cumulative_loss_protected_npv = 0
    cumulative_maintenance_npv = 0
    for i, year in enumerate(YEARS):
        annual_revenue = BASE_ECONOMIC_VALUE * ((1 + GROWTH_RATE) ** i)
        discount_factor = 1 / ((1 + DISCOUNT_RATE) ** i)
        pv_revenue = annual_revenue * discount_factor
        cumulative_econ_npv += pv_revenue
        t_from_1990 = 35 + i
        lambda_freq = get_weather_frequency(t_from_1990)
        eal_unprotected = lambda_freq * AVG_LOSS_PER_EVENT_UNPROTECTED
        eal_protected = eal_unprotected * (1 - RISK_REDUCTION_RATE)
        pv_loss_unprotected = eal_unprotected * discount_factor
        cumulative_loss_unprotected_npv += pv_loss_unprotected
        pv_loss_protected = eal_protected * discount_factor
        cumulative_loss_protected_npv += pv_loss_protected
        pv_maintenance = COST_MAINTENANCE_YEARLY * discount_factor
        cumulative_maintenance_npv += pv_maintenance
        simulation_data.append({
            'Year': year,
            'Lambda(Freq)': lambda_freq,
            'EAL_Unprotected': eal_unprotected,
            'EAL_Protected': eal_protected
        })
        
    df_sim = pd.DataFrame(simulation_data)

    V_econ = cumulative_econ_npv
    V_culture_monetized = target_score * ALPHA_FACTOR
    TV = V_econ + V_culture_monetized
    TC = COST_INITIAL_MEASURE + cumulative_maintenance_npv + cumulative_loss_protected_npv
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

df_result = run_preservation_model()