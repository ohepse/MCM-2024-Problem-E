import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

class PropertyResilienceModel:
    def __init__(self, name, params):
        self.name = name
        # --- 模型参数 ---
        self.T = params.get('T', 20)          # 规划周期 (年)
        self.V_base = params.get('V_base', 100) # 基础建筑价值 (标准化为100)
        self.beta = params.get('beta', 0.05)    # 建设成本敏感系数 (h^2的系数)
        self.gamma = params.get('gamma', 0.8)   # 韧性减灾系数 (指数衰减)
        self.lambda_0 = params.get('lambda_0', 0.2) # 初始灾害频率 (次/年)
        self.k = params.get('k', 0.05)          # 气候恶化因子 (频率年增长率)
        self.g = params.get('g', 0.02)          # 资产增值/通胀率
        self.D_max = params.get('D_max', 1.0)   # 最大易损性 (0-1)
        self.rho = params.get('rho', 0.2)       # 保险公司利润率
        self.income_limit = params.get('income_limit', 500) # 当地平均收入(用于可行性判定)

    def construction_cost(self, h):
        """计算初始建设成本 C_build(h) = V_base * (1 + beta * h^2)"""
        return self.V_base * (1 + self.beta * h**2)

    def damage_ratio(self, h):
        """计算易损性 D(h) = D_max * e^(-gamma * h)"""
        return self.D_max * np.exp(-self.gamma * h)

    def climate_risk_integral(self):
        """计算时间累积风险因子 Omega = (e^(KT) - 1) / K"""
        K = self.k + self.g
        if K == 0:
            return self.T
        return (np.exp(K * self.T) - 1) / K

    def total_lifecycle_cost(self, h):
        """计算全生命周期总成本 Z(h)"""
        # 1. 初始建设成本
        C_build = self.construction_cost(h)

        # 2. 未来总风险成本 (积分形式)
        # 公式: RiskCost = (1+rho) * lambda_0 * C_build * D(h) * Omega
        omega = self.climate_risk_integral()
        
        # 风险乘数
        risk_multiplier = (1 + self.rho) * self.lambda_0 * self.damage_ratio(h) * omega
        
        # 总风险成本 (包含了未来所有年份的保费 + 自留风险)
        total_risk_cost = C_build * risk_multiplier
        
        return C_build + total_risk_cost

    def optimize(self):
        """求解最优 h*，使 Z(h) 最小"""
        # 限制 h 在 0 到 10 之间
        res = minimize_scalar(self.total_lifecycle_cost, bounds=(0, 10), method='bounded')
        return res.x, res.fun

    def check_feasibility(self, h_opt):
        """【Whether】判定: 检查第10年的保费是否可负担"""
        t_check = 10
        C_build = self.construction_cost(h_opt)
        D_h = self.damage_ratio(h_opt)
        
        # 计算第10年的瞬时风险保费
        lambda_t = self.lambda_0 * np.exp(self.k * t_check)
        value_t = C_build * (1 + self.g)**t_check
        
        premium_y10 = (1 + self.rho) * lambda_t * value_t * D_h
        
        # 判定标准: 保费 < 收入的 5%
        is_feasible = premium_y10 < (self.income_limit * 0.05)
        return is_feasible, premium_y10

# --- 场景设置 (Scenario Setup) ---

# 场景 A: "风暴海岸" (高风险, 气候恶化快)
# lambda_0=0.5 (两年一遇), k=0.06 (每年恶化6%)
params_A = {
    'T': 20, 'V_base': 100, 'beta': 0.05, 'gamma': 0.8,
    'lambda_0': 0.5, 'k': 0.06, 'g': 0.02, 'rho': 0.2,
    'income_limit': 1000
}

# 场景 B: "宁静山谷" (低风险, 气候稳定)
# lambda_0=0.1 (十年一遇), k=0.01 (每年恶化1%)
params_B = {
    'T': 20, 'V_base': 100, 'beta': 0.05, 'gamma': 0.8,
    'lambda_0': 0.1, 'k': 0.01, 'g': 0.02, 'rho': 0.2,
    'income_limit': 800
}

model_A = PropertyResilienceModel("Scenario A (High Risk)", params_A)
model_B = PropertyResilienceModel("Scenario B (Low Risk)", params_B)

# --- 运行优化 ---
h_opt_A, cost_A = model_A.optimize()
h_opt_B, cost_B = model_B.optimize()
feasible_A, premium_A = model_A.check_feasibility(h_opt_A)
feasible_B, premium_B = model_B.check_feasibility(h_opt_B)

# --- 绘图 (Plotting) ---
h_values = np.linspace(0, 8, 100)

def get_curves(model):
    costs_build = [model.construction_cost(h) for h in h_values]
    costs_total = [model.total_lifecycle_cost(h) for h in h_values]
    costs_risk = [t - b for t, b in zip(costs_total, costs_build)]
    return costs_build, costs_risk, costs_total

build_A, risk_A, total_A = get_curves(model_A)
build_B, risk_B, total_B = get_curves(model_B)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 绘制场景 A
ax1.plot(h_values, build_A, 'g--', label='Construction Cost')
ax1.plot(h_values, risk_A, 'r--', label='Risk/Insurance Cost')
ax1.plot(h_values, total_A, 'b-', linewidth=2, label='Total Lifecycle Cost (Z)')
ax1.scatter([h_opt_A], [cost_A], color='black', zorder=5)
ax1.annotate(f'Optimal $h^*={h_opt_A:.2f}$', (h_opt_A, cost_A), xytext=(h_opt_A, cost_A+200), arrowprops=dict(arrowstyle='->'))
ax1.set_title(f"Scenario A: High Risk ($k={params_A['k']}$)")
ax1.set_xlabel('Resilience Level ($h$)')
ax1.set_ylabel('Cost')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 绘制场景 B
ax2.plot(h_values, build_B, 'g--', label='Construction Cost')
ax2.plot(h_values, risk_B, 'r--', label='Risk/Insurance Cost')
ax2.plot(h_values, total_B, 'b-', linewidth=2, label='Total Lifecycle Cost (Z)')
ax2.scatter([h_opt_B], [cost_B], color='black', zorder=5)
ax2.annotate(f'Optimal $h^*={h_opt_B:.2f}$', (h_opt_B, cost_B), xytext=(h_opt_B+0, cost_B+50), arrowprops=dict(arrowstyle='->'))
ax2.set_title(f"Scenario B: Low Risk ($k={params_B['k']}$)")
ax2.set_xlabel('Resilience Level ($h$)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# # --- 准备参数 ---
# # 锁定其他参数，只改变 k 或 beta
# base_params = {
#     'T': 20, 'V_base': 100, 'gamma': 0.8,
#     'lambda_0': 0.2, 'g': 0.02, 'rho': 0.2, 'D_max': 1.0,
#     'income_limit': 800
# }

# # --- 分析 1: 单变量敏感性 (k 对 h* 的影响) ---
# k_values = np.linspace(0, 0.15, 50) # 测试 k 从 0% 到 15%
# optimal_h_list = []
# total_cost_list = []

# model = PropertyResilienceModel("Sensitivity Test", base_params)

# for k_val in k_values:
#     model.k = k_val # 动态修改 k
#     h_opt, cost = model.optimize()
#     optimal_h_list.append(h_opt)

# # --- 分析 2: 双变量可行性边界 (k vs beta) ---
# beta_values = np.linspace(0.01, 1, 50) # 测试造价系数
# k_grid = np.linspace(0, 1, 50)

# feasible_grid = np.zeros((len(beta_values), len(k_grid)))

# for i, b_val in enumerate(beta_values):
#     for j, k_val in enumerate(k_grid):
#         model.beta = b_val
#         model.k = k_val
        
#         # 1. 先找到该情况下的最优 h*
#         h_opt, _ = model.optimize()
        
#         # 2. 检查该 h* 是否可行
#         is_feasible, _ = model.check_feasibility(h_opt)
        
#         # 3. 存入矩阵 (1=可行, 0=不可行)
#         feasible_grid[i, j] = 1 if is_feasible else 0

# # --- 绘图 ---
# fig,  ax2 = plt.subplots(1, 1, figsize=(8, 8))

# # 图 2: 可行性边界 (Phase Diagram)
# # 使用 contourf 画出区域
# X, Y = np.meshgrid(k_grid, beta_values)
# contour = ax2.contourf(X, Y, feasible_grid, levels=[-0.1, 0.5, 1.1], colors=['#ffcccc', '#ccffcc'])
# # 添加图例: 红色=不可行, 绿色=可行
# proxies = [plt.Rectangle((0,0),1,1,fc = '#ccffcc'), plt.Rectangle((0,0),1,1,fc = '#ffcccc')]
# ax2.legend(proxies, ["Feasible Region", "Unfeasible Region"])
# ax2.axvspan(0.03, 0.06, color='blue', alpha=0.1, hatch='//') # Hatch to be visible over colors
# ax2.axvline(0.03, color='blue', linestyle='--')
# ax2.axvline(0.06, color='blue', linestyle='--')
# # Label
# ax2.text(0.13, 0.18, "Current\nProjection\n(2025-2050)", color='blue', ha='center', fontweight='bold', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
# ax2.set_title("Stability Analysis: Feasibility Boundary")
# ax2.set_xlabel("Climate Deterioration Rate ($k$)")
# ax2.set_ylabel("Construction Cost Factor ($\\beta$)")

# plt.tight_layout()
# plt.show()