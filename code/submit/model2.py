import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

class PropertyResilienceModel:
    def __init__(self, name, params):
        self.name = name
        # --- 模型参数 ---
        self.T = params.get('T', 20)          
        self.V_base = params.get('V_base', 100) 
        self.beta = params.get('beta', 0.05)    
        self.gamma = params.get('gamma', 0.8)   
        self.lambda_0 = params.get('lambda_0', 0.2) 
        self.k = params.get('k', 0.05)         
        self.g = params.get('g', 0.02)          
        self.D_max = params.get('D_max', 1.0)   
        self.rho = params.get('rho', 0.2)       
        self.income_limit = params.get('income_limit', 500) 
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

params_A = {
    'T': 20, 'V_base': 100, 'beta': 0.05, 'gamma': 0.8,
    'lambda_0': 0.5, 'k': 0.06, 'g': 0.02, 'rho': 0.2,
    'income_limit': 1000
}
params_B = {
    'T': 20, 'V_base': 100, 'beta': 0.05, 'gamma': 0.8,
    'lambda_0': 0.1, 'k': 0.01, 'g': 0.02, 'rho': 0.2,
    'income_limit': 800
}
model_A = PropertyResilienceModel("Scenario A (High Risk)", params_A)
model_B = PropertyResilienceModel("Scenario B (Low Risk)", params_B)
h_opt_A, cost_A = model_A.optimize()
h_opt_B, cost_B = model_B.optimize()
feasible_A, premium_A = model_A.check_feasibility(h_opt_A)
feasible_B, premium_B = model_B.check_feasibility(h_opt_B)



