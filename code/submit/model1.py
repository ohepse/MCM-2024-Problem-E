import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

path_freq = r'C:\Users\mth\Desktop\20250124\code\disaster_frequency.csv'
path_loss = r'C:\Users\mth\Desktop\20250124\code\loss_severity.csv'
path_econ = r'C:\Users\mth\Desktop\20250124\code\economic_data.csv'
df_freq = pd.read_csv(path_freq)
years = df_freq['Year'].values
params = {}
for region in ['Texas_USA', 'Luzon_PHL']:
    counts = df_freq[region].values
    a, k = fit_nhpp_model(years, counts, region)
    params[region] = {'a': a, 'k': k}
    print(f"✅ {region} result: a={a:.2f}, k={k:.4f}")
df_loss = pd.read_csv(path_loss)
median_severity = df_loss.groupby('Region')['Total_Loss_000_USD'].median()
mean_severity = df_loss.groupby('Region')['Total_Loss_000_USD'].mean()
L0_dict = {}
for region in ['Texas_USA', 'Luzon_PHL']:
    if region in median_severity:
        raw_L0 = 0.7 * median_severity[region] + 0.3 * mean_severity[region]
    else:
        raw_L0 = 100000 
    L0_dict[region] = raw_L0

df_econ = pd.read_csv(path_econ)
future_years = np.arange(1990, 2060)
t_future = future_years - 1990
current_idx = 2024 - 1990
MITIGATION_EFFICIENCY = 0.20
params_setting = {
    'Texas_USA': {'coverage_ratio': 0.50, 'profit_margin': 0.15, 'burden_start': 0.6},
    'Luzon_PHL': {'coverage_ratio': 0.10, 'profit_margin': 0.20, 'burden_start': 0.45}
}
results = {}
for region in ['Texas_USA', 'Luzon_PHL']:
    a = params[region]['a']
    k = params[region]['k']
    raw_L0 = L0_dict[region]
    settings = params_setting[region]
    last_econ = df_econ.iloc[-1]
    g_inc = last_econ['Texas_Growth'] if 'Texas' in region else last_econ['Luzon_Growth']
    lambda_t = a * np.exp(k * t_future)
    g_asset = g_inc + 0.01 
    insurance_L0 = raw_L0 * settings['coverage_ratio']
    expected_loss_t = insurance_L0 * ((1 + g_asset) ** t_future)
    p_min_curve = (1 + settings['profit_margin']) * lambda_t * expected_loss_t
    p_mitigated_curve = p_min_curve * (1 - MITIGATION_EFFICIENCY)
    current_cost = p_min_curve[current_idx]
    current_limit = current_cost / settings['burden_start']
    p_max_curve = current_limit * ((1 + g_inc) ** (t_future - current_idx))
    diff_orig = p_min_curve - p_max_curve
    break_orig = np.where((diff_orig > 0) & (future_years > 2025))[0]
    year_orig = future_years[break_orig[0]] if len(break_orig) > 0 else None
    diff_mit = p_mitigated_curve - p_max_curve
    break_mit = np.where((diff_mit > 0) & (future_years > 2025))[0]
    year_mit = future_years[break_mit[0]] if len(break_mit) > 0 else None
    results[region] = {
        'years': future_years,
        'p_min': p_min_curve,
        'p_mitigated': p_mitigated_curve,
        'p_max': p_max_curve,
        'crash_orig': year_orig,
        'crash_mit': year_mit
    }


