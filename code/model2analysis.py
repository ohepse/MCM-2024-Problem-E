import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

class PropertyResilienceModel:
    def __init__(self, name, params):
        self.name = name
        self.T = params.get('T', 20)
        self.V_base = params.get('V_base', 100)
        self.beta = params.get('beta', 0.05)
        self.gamma = params.get('gamma', 0.8)
        self.lambda_0 = params.get('lambda_0', 0.2)
        self.k = params.get('k', 0.05)
        self.g = params.get('g', 0.02)
        self.D_max = params.get('D_max', 1.0)
        self.rho = params.get('rho', 0.2)
        self.income_limit = params.get('income_limit', 800)

    def construction_cost(self, h):
        return self.V_base * (1 + self.beta * h**2)

    def damage_ratio(self, h):
        return self.D_max * np.exp(-self.gamma * h)

    def climate_risk_integral(self):
        K = self.k + self.g
        if K == 0:
            return self.T
        return (np.exp(K * self.T) - 1) / K

    def total_lifecycle_cost(self, h):
        C_build = self.construction_cost(h)
        omega = self.climate_risk_integral()
        risk_multiplier = (1 + self.rho) * self.lambda_0 * self.damage_ratio(h) * omega
        total_risk_cost = C_build * risk_multiplier
        return C_build + total_risk_cost

    def optimize(self):
        res = minimize_scalar(self.total_lifecycle_cost, bounds=(0, 10), method='bounded')
        return res.x, res.fun

    def check_feasibility(self, h_opt):
        t_check = 10
        C_build = self.construction_cost(h_opt)
        D_h = self.damage_ratio(h_opt)
        lambda_t = self.lambda_0 * np.exp(self.k * t_check)
        value_t = C_build * (1 + self.g)**t_check
        premium_y10 = (1 + self.rho) * lambda_t * value_t * D_h
        is_feasible = premium_y10 < (self.income_limit * 0.05)
        return is_feasible, premium_y10

# Base parameters
base_params = {
    'T': 20, 'V_base': 100, 'gamma': 0.8,
    'lambda_0': 0.2, 'g': 0.02, 'rho': 0.2, 'D_max': 1.0,
    'income_limit': 800,
    'k': 0.05, 'beta': 0.05
}

# --- Analysis 1: Beta vs h* (New Request) ---
beta_values = np.linspace(0.01, 0.20, 50)
h_opt_beta = []
model = PropertyResilienceModel("Beta Sensitivity", base_params)

for b in beta_values:
    model.beta = b
    # Reset k to baseline for this analysis
    model.k = 0.05
    h, _ = model.optimize()
    h_opt_beta.append(h)

# --- Analysis 2: k vs h* (For context/annotation) ---
k_values = np.linspace(0, 0.15, 50)
h_opt_k = []
model.beta = 0.05 # Reset beta
for k_val in k_values:
    model.k = k_val
    h, _ = model.optimize()
    h_opt_k.append(h)

# --- Analysis 3: Feasibility Boundary (k vs beta) ---
# Re-run to add the annotation
k_grid = np.linspace(0, 0.15, 50)
beta_grid = np.linspace(0.01, 0.20, 50)
feasible_grid = np.zeros((len(beta_grid), len(k_grid)))

for i, b_val in enumerate(beta_grid):
    for j, k_val in enumerate(k_grid):
        model.beta = b_val
        model.k = k_val
        h, _ = model.optimize()
        is_feasible, _ = model.check_feasibility(h)
        feasible_grid[i, j] = 1 if is_feasible else 0

# --- Plotting ---
fig = plt.figure(figsize=(13, 6))

# Plot 1: Beta vs h*
ax1 = plt.subplot(1, 2, 1)
ax1.plot(beta_values, h_opt_beta, 'purple', linewidth=2)
ax1.set_title("Cost Sensitivity: Optimal Resilience ($h^*$) vs. Cost Factor ($\\beta$)")
ax1.set_xlabel("Construction Cost Factor ($\\beta$)")
ax1.set_ylabel("Optimal Resilience Level ($h^*$)")
ax1.grid(True, alpha=0.3)
ax1.annotate('As building gets expensive,\noptimal resilience drops',
             xy=(0.15, h_opt_beta[-1]+0.3), xytext=(0.1, h_opt_beta[-1]+1.2),fontweight = 'bold',
             arrowprops=dict(facecolor='black', shrink=0.05))

# Plot 2: k vs h* with Annotation
ax2 = plt.subplot(1, 2, 2)
ax2.plot(k_values, h_opt_k, 'b-', linewidth=2)
# Add Annotation
ax2.axvspan(0.03, 0.06, color='orange', alpha=0.2, label='Current Projection (2025-2050)')
ax2.axvline(0.03, color='orange', linestyle='--')
ax2.axvline(0.06, color='orange', linestyle='--')
ax2.set_title("Climate Sensitivity: $h^*$ vs. Climate Rate ($k$)")
ax2.set_xlabel("Climate Deterioration Rate ($k$)")
ax2.set_ylabel("Optimal Resilience Level ($h^*$)")
ax2.legend()
ax2.grid(True, alpha=0.3)



plt.tight_layout()
plt.show()
