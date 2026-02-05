import numpy as np
import matplotlib.pyplot as plt

# 设置绘图风格
plt.style.use('ggplot')

# ---------------------------------------------------------
# 1. 定义基准参数 (Baseline Parameters)
# ---------------------------------------------------------
# 根据论文描述，调整参数以使得 Texas 在 ~2048 崩溃，Luzon 在 ~2057 崩溃
params_texas = {
    'a': 5.0,             # 基础灾害频率
    'k': 0.04,            # 气候恶化因子 (核心风险驱动)
    'raw_L0': 100000,     # 单次灾害基础损失
    'coverage_ratio': 0.50, # 保险覆盖率
    'profit_margin': 0.15,  # 保险公司利润率
    'burden_start': 0.6,    # 初始保费负担 (Current Cost / Max Limit)
    'g_inc': 0.02           # 收入增长率
}

params_luzon = {
    'a': 8.0,
    'k': 0.035,
    'raw_L0': 50000,
    'coverage_ratio': 0.10, # 低覆盖率掩盖了部分风险
    'profit_margin': 0.20,
    'burden_start': 0.45,   # 初始负担较低
    'g_inc': 0.03           # 较高的经济增长率
}

regions = {'Texas_USA': params_texas, 'Luzon_PHL': params_luzon}

# ---------------------------------------------------------
# 2. 定义核心计算函数
# ---------------------------------------------------------
def get_p_max_curve(region_params, alpha_multiplier=1.0):
    """
    计算最大可负担保费曲线 (P_max)
    alpha_multiplier 用于模拟支付能力的变化
    """
    start_year, end_year = 1990, 2080
    current_year = 2024
    future_years = np.arange(start_year, end_year)
    t_future = future_years - start_year
    
    # 在当前年份校准 P_max
    # 计算当前年份的理论保费 P_min(current)
    lambda_t_curr = region_params['a'] * np.exp(region_params['k'] * (current_year - start_year))
    g_asset = region_params['g_inc'] + 0.01
    loss_curr = (region_params['raw_L0'] * region_params['coverage_ratio']) * \
                ((1 + g_asset) ** (current_year - start_year))
    p_min_curr = (1 + region_params['profit_margin']) * lambda_t_curr * loss_curr
    
    # P_max(current) = P_min(current) / burden_start
    # alpha_multiplier > 1 代表支付能力提升
    current_limit = (p_min_curr / region_params['burden_start']) * alpha_multiplier
    
    # 随收入增长推演未来 P_max
    p_max_curve = current_limit * ((1 + region_params['g_inc']) ** (t_future - (current_year - start_year)))
    return future_years, p_max_curve

def get_collapse_year(p_min_curve, p_max_curve, years):
    """
    寻找市场崩溃年份 (P_min > P_max 的最早年份)
    """
    diff = p_min_curve - p_max_curve
    # 只考虑 2025 年以后的崩溃
    future_mask = years > 2025
    break_indices = np.where(future_mask & (diff > 0))[0]
    
    if len(break_indices) > 0:
        return years[break_indices[0]]
    else:
        return years[-1] # 在模拟范围内未崩溃

# ---------------------------------------------------------
# 3. 执行敏感性分析
# ---------------------------------------------------------

# 分析 1: 气候恶化因子 (k)
k_multipliers = np.linspace(0.5, 1.5, 20) # 在基准值的 50% 到 150% 之间波动
results_k = {r: [] for r in regions}

for region_name, params in regions.items():
    years, p_max = get_p_max_curve(params) # P_max 固定
    for km in k_multipliers:
        k_val = params['k'] * km
        # 重新计算 P_min
        t_future = years - 1990
        lambda_t = params['a'] * np.exp(k_val * t_future)
        g_asset = params['g_inc'] + 0.01
        loss_t = (params['raw_L0'] * params['coverage_ratio']) * ((1 + g_asset) ** t_future)
        p_min = (1 + params['profit_margin']) * lambda_t * loss_t
        
        results_k[region_name].append(get_collapse_year(p_min, p_max, years))

# 分析 2: 减灾有效性 (eta)
eta_values = np.linspace(0.0, 0.5, 20) # 减损率从 0% 到 50%
results_eta = {r: [] for r in regions}

for region_name, params in regions.items():
    years, p_max = get_p_max_curve(params)
    # 计算基准 P_min (未减灾)
    t_future = years - 1990
    lambda_t = params['a'] * np.exp(params['k'] * t_future)
    g_asset = params['g_inc'] + 0.01
    loss_t = (params['raw_L0'] * params['coverage_ratio']) * ((1 + g_asset) ** t_future)
    p_min_base = (1 + params['profit_margin']) * lambda_t * loss_t
    
    for eta in eta_values:
        p_min_mitigated = p_min_base * (1 - eta)
        results_eta[region_name].append(get_collapse_year(p_min_mitigated, p_max, years))

# 分析 3: 支付能力 (alpha)
alpha_multipliers = np.linspace(0.8, 1.2, 20) # 支付能力在基准的 +/- 20% 波动
results_alpha = {r: [] for r in regions}

for region_name, params in regions.items():
    # 计算基准 P_min
    t_future = years - 1990
    lambda_t = params['a'] * np.exp(params['k'] * t_future)
    g_asset = params['g_inc'] + 0.01
    loss_t = (params['raw_L0'] * params['coverage_ratio']) * ((1 + g_asset) ** t_future)
    p_min = (1 + params['profit_margin']) * lambda_t * loss_t
    
    for am in alpha_multipliers:
        _, p_max_varied = get_p_max_curve(params, alpha_multiplier=am)
        results_alpha[region_name].append(get_collapse_year(p_min, p_max_varied, years))

# ---------------------------------------------------------
# 4. 绘图
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Plot 1: Climate k
ax = axes[0]
for region in regions:
    ax.plot(k_multipliers, results_k[region], marker='o', label=region)
ax.set_title('Sensitivity to Climate Deterioration (k)')
ax.set_xlabel('Multiplier of Base k')
ax.set_ylabel('Market Collapse Year')
ax.legend()
ax.grid(True)

# Plot 2: Mitigation Eta
ax = axes[1]
for region in regions:
    ax.plot(eta_values, results_eta[region], marker='s', label=region)
ax.set_title('Sensitivity to Mitigation (η)')
ax.set_xlabel('Mitigation Efficiency (Loss Reduction)')
ax.set_ylabel('Market Collapse Year')
ax.legend()
ax.grid(True)


plt.tight_layout()
plt.show()