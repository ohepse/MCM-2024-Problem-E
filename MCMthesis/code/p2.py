import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel("ath.xlsx")
countries = df['Country'].values
X_raw = df[['100m', '200m', '400m', '800m', '1500m', '5000m', '10000m', 'marathon']].values

scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)


fa = FactorAnalyzer(n_factors=2, rotation='varimax', method='minres')
fa.fit(X_std)
loadings = fa.loadings_
factor_scores = fa.transform(X_std)


sprint_avg_f1 = loadings[[0, 1, 2], 0].mean()
sprint_avg_f2 = loadings[[0, 1, 2], 1].mean()

if sprint_avg_f1 > sprint_avg_f2:
    sprint_factor = -factor_scores[:, 0]
    endurance_factor = -actor_scores[:, 1]
else:
    sprint_factor = -factor_scores[:, 1]
    endurance_factor = -factor_scores[:, 0]

x_min, x_max = min(sprint_factor) - 0.5, max(sprint_factor) + 0.5
y_min, y_max = min(endurance_factor) - 0.5, max(endurance_factor) + 0.5

plt.figure(figsize=(8, 8))

plt.scatter(sprint_factor, endurance_factor, 
            color='#1f77b4', alpha=0.8, s=80, edgecolors='black', linewidth=0.5)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel('爆发力因子得分', fontsize=12)
plt.ylabel('耐力因子得分', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()