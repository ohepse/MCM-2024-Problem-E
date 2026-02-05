import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use('ggplot')


def fit_nhpp_model(years, counts, region_name):
    def exponential_func(t, a, k):
        return a * np.exp(k * t)

    t_data = years - years.min()
    try:
        popt, pcov = curve_fit(exponential_func, t_data, counts, p0=[1, 0.01], maxfev=5000)
        a_fit, k_fit = popt
    except:
        print(f"⚠️ {region_name} 拟合失败，使用默认参数")
        a_fit, k_fit = np.mean(counts), 0.01
    return a_fit, k_fit

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
    print(f"✅ {region} 拟合结果: a={a:.2f}, k={k:.4f}")
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


fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, region in zip(axes, ['Texas_USA', 'Luzon_PHL']):
    res = results[region]
    ax.plot(res['years'], res['p_min'], 'r-', label='Original Risk Cost', linewidth=2, alpha=0.6)
    ax.plot(res['years'], res['p_mitigated'], 'g-', label='Mitigated Cost (Actions Taken)', linewidth=2.5)
    ax.plot(res['years'], res['p_max'], 'b--', label='Affordability Limit', linewidth=2)
    ax.axvline(x=2024, color='gray', linestyle=':', alpha=0.5)
    if res['crash_orig']:
        y_val_orig = res['p_min'][res['years'] == res['crash_orig']][0]
        ax.plot(res['crash_orig'], y_val_orig, 'ro', markersize=8)
        ax.annotate(f"Market Fail\n{res['crash_orig']}", 
                    xy=(res['crash_orig'], y_val_orig),         
                    xytext=(-50, 40),                           
                    textcoords='offset points',
                    arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8), 
                    color='red', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)) 
    if res['crash_mit']:
        y_val_mit = res['p_mitigated'][res['years'] == res['crash_mit']][0]
        
        ax.plot(res['crash_mit'], y_val_mit, 'go', markersize=10)
        ax.annotate(f"Extended to\n{res['crash_mit']}", 
                    xy=(res['crash_mit'], y_val_mit),
                    xytext=(20, -30), textcoords='offset points',
                    arrowprops=dict(facecolor='green', shrink=0.05), 
                    color='green', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
        if res['crash_orig'] and res['crash_mit'] > res['crash_orig']:
             ax.axvspan(res['crash_orig'], res['crash_mit'], color='green', alpha=0.1, label='Time Gained')
    ax.set_title(f"{region}: Impact of Mitigation")
    ax.set_xlabel("Year")
    ax.set_ylabel("Financial Scale (Normalized)")
    ax.legend()
    ax.set_xlim(2010, 2060)

plt.tight_layout()
plt.show()
