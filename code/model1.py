import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 设置绘图风格，让图表更好看
plt.style.use('ggplot')

# ==========================================
# 第一步：拟合灾害频率模型 (NHPP)
# 目标：找出 lambda(t) = a * exp(k * t) 中的参数 a 和 k
# ==========================================

def fit_nhpp_model(years, counts, region_name):
    # 定义指数增长函数
    def exponential_func(t, a, k):
        return a * np.exp(k * t)

    # 将年份归一化 (从 t=0 开始)，方便计算
    t_data = years - years.min()
    
    # 拟合参数
    # p0 是初始猜测值 [a=1, k=0.01]
    try:
        popt, pcov = curve_fit(exponential_func, t_data, counts, p0=[1, 0.01], maxfev=5000)
        a_fit, k_fit = popt
    except:
        print(f"⚠️ {region_name} 拟合失败，使用默认参数")
        a_fit, k_fit = np.mean(counts), 0.01

    return a_fit, k_fit

# 读取频率数据
df_freq = pd.read_csv(r'C:\Users\mth\Desktop\20250124\code\disaster_frequency.csv')
years = df_freq['Year'].values

# 分别拟合 Texas 和 Luzon
params = {}
for region in ['Texas_USA', 'Luzon_PHL']:
    counts = df_freq[region].values
    a, k = fit_nhpp_model(years, counts, region)
    params[region] = {'a': a, 'k': k}
    print(f"✅ {region} 拟合结果: 初始频率 a={a:.2f}, 恶化因子 k={k:.4f} ({(np.exp(k)-1)*100:.2f}%/年)")

# ==========================================
# 第二步：计算单次灾害的平均损失 (Severity)
# ==========================================

df_loss = pd.read_csv(r'C:\Users\mth\Desktop\20250124\code\loss_severity.csv')

# 计算每个地区的平均单次损失 (Expected Loss per Event)
avg_severity = df_loss.groupby('Region')['Total_Loss_000_USD'].mean()

# 如果某个地区没有损失数据（极端情况），给一个默认值
for region in ['Texas_USA', 'Luzon_PHL']:
    if region not in avg_severity:
        avg_severity[region] = 100000 # 默认 1亿美元
    print(f"💰 {region} 平均单次损失: ${avg_severity[region]/1000:.2f} Million")

# ==========================================
# 优化后的第三步：决策模型 (引入修正因子)
# ==========================================

df_econ = pd.read_csv(r'C:\Users\mth\Desktop\20250124\code\economic_data.csv')
future_years = np.arange(1990, 2060) # 预测到 2060 年
t_future = future_years - 1990

# 设定模型假设参数
PROFIT_MARGIN = 0.20      # 保险公司利润率 + 运营成本 (20%)
AFFORDABILITY_RATIO = 0.05 # 家庭能拿出收入的 5% 买保险


# 1. 计算损失基准 (L0) - 使用中位数而非平均值，排除极端值干扰
# ---------------------------------------------------------
# 计算中位数
median_severity = df_loss.groupby('Region')['Total_Loss_000_USD'].median()
# 为了防止中位数过小（比如有很多小灾害），我们可以取 Mean 和 Median 的加权平均
# 或者直接用 log-normal 分布的期望值（更高级，但大一可以用简单加权）
mean_severity = df_loss.groupby('Region')['Total_Loss_000_USD'].mean()

L0_dict = {}
for region in ['Texas_USA', 'Luzon_PHL']:
    # 综合考量：70%权重给中位数，30%给平均值
    if region in median_severity:
        raw_L0 = 0.7 * median_severity[region] + 0.3 * mean_severity[region]
    else:
        raw_L0 = 100000 # 默认值
    L0_dict[region] = raw_L0

# 2. 设定模型参数 (Model Parameters)
# ---------------------------------------------------------
params_setting = {
    'Texas_USA': {
        'coverage_ratio': 0.50,  # 发达国家：约50%损失由保险覆盖
        'profit_margin': 0.15,   # 利润率
        'burden_start': 0.6     # 当前保费占支付能力的 35% (市场健康)
    },
    'Luzon_PHL': {
        'coverage_ratio': 0.10,  # 发展中国家：仅10%有保险 (Protection Gap 巨大)
        'profit_margin': 0.20,   # 风险更高，要求利润更高
        'burden_start': 0.45     # 收入低，当前保险渗透率极低
    }
}

future_years = np.arange(1990, 2060)
t_future = future_years - 1990
current_idx = 2024 - 1990

results = {}

for region in ['Texas_USA', 'Luzon_PHL']:
    # 读取 NHPP 参数
    a = params[region]['a']
    k = params[region]['k']
    
    # 读取修正后的基准损失
    raw_L0 = L0_dict[region]
    
    # 读取特定地区的设定
    settings = params_setting[region]
    insurance_L0 = raw_L0 * settings['coverage_ratio'] # 只计算保险赔付的部分
    
    # 读取经济增长率
    last_econ = df_econ.iloc[-1]
    if 'Texas' in region:
        g_inc = last_econ['Texas_Growth']
    else:
        g_inc = last_econ['Luzon_Growth']
    
    # --- 计算 P_min (保险公司成本) ---
    # 公式：Rate * Freq * Insured_Severity
    lambda_t = a * np.exp(k * t_future)
    
    # 假设资产价值随收入增长 (g_inc) 加上 1% 的沿海资产溢价
    g_asset = g_inc + 0.01 
    expected_loss_t = insurance_L0 * ((1 + g_asset) ** t_future)
    
    p_min_curve = (1 + settings['profit_margin']) * lambda_t * expected_loss_t
    
    # --- 计算 P_max (支付能力上限) ---
    # 核心逻辑：我们不需要知道具体的美元金额，只需要知道相对趋势
    # 设定 2024 年的 P_max 是当前 P_min 的 X 倍
    
    current_cost = p_min_curve[current_idx]
    # 如果 burden_start = 0.35，说明 P_max = Cost / 0.35
    current_limit = current_cost / settings['burden_start']
    
    # 上限随收入增长
    p_max_curve = current_limit * ((1 + g_inc) ** (t_future - current_idx))

    # --- 寻找交点 ---
    diff = p_min_curve - p_max_curve
    # 只看 2025 以后的交点
    future_break_even = np.where((diff > 0) & (future_years > 2025))[0]
    
    if len(future_break_even) > 0:
        crash_year = future_years[future_break_even[0]]
    else:
        crash_year = None
        
    results[region] = {
        'years': future_years,
        'p_min': p_min_curve,
        'p_max': p_max_curve,
        'crash_year': crash_year,
        'k': k
    }

# 打印新结果
print("-" * 50)
for region, res in results.items():
    print(f"🌍 地区: {region}")
    print(f"   📊 保险覆盖率设定: {params_setting[region]['coverage_ratio']*100}%")
    if res['crash_year']:
        print(f"   ⚠️ 预计市场崩溃年份: {res['crash_year']} 年")
    else:
        print(f"   ✅ 2060年前保持可持续")
print("-" * 50)

# ==========================================
# 第四步：可视化结果
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, region in zip(axes, ['Texas_USA', 'Luzon_PHL']):
    res = results[region]
    
    # 绘制曲线
    ax.plot(res['years'], res['p_min'], 'r-', label='Insurance Cost (Risk)', linewidth=2)
    ax.plot(res['years'], res['p_max'], 'b--', label='Affordability Limit', linewidth=2)
    
    # 标记当前年份
    ax.axvline(x=2024, color='gray', linestyle=':', alpha=0.5)
    
    # 标记崩溃年份
    if res['crash_year']:
        ax.plot(res['crash_year'], res['p_min'][res['years'] == res['crash_year']], 'ko', markersize=10)
        ax.annotate(f'Market Failure\n{res["crash_year"]}', 
                    xy=(res['crash_year'], res['p_min'][res['years'] == res['crash_year']]),
                    xytext=(-60, 40), textcoords='offset points',
                    arrowprops=dict(facecolor='black', shrink=0.05))
        ax.axvspan(res['crash_year'], 2060, color='red', alpha=0.1, label='Uninsurable Zone')
    
    ax.set_title(f"{region} Insurance Sustainability")
    ax.set_xlabel("Year")
    ax.set_ylabel("Financial Scale (Normalized)")
    ax.legend()
    ax.set_xlim(2000, 2060)

plt.tight_layout()
plt.show()

print("\n分析完成！请查看弹出的图表。")
for region, res in results.items():
    if res['crash_year']:
        print(f"⚠️ 警告: {region} 预计将在 {res['crash_year']} 年达到不可保临界点。")
    else:
        print(f"✅ {region} 在 2060 年前保持可持续。")