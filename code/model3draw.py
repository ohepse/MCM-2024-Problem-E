import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


YEARS = np.arange(2025, 2051)
DISCOUNT_RATE = 0.04
BASE_REVENUE = 171e6          
GROWTH_RATE = 0.02
COST_INITIAL = 550e6          
COST_MAINTENANCE_YEARLY = 3e6 

cumulative_net_cashflow = -COST_INITIAL
cum_econ_val = 0
plot_data = []

def get_lambda(year):
    t = year - 1990
    return 5.87 * np.exp(0.0122 * t)

for i, year in enumerate(YEARS):
    revenue = BASE_REVENUE * ((1 + GROWTH_RATE) ** i)
    maintenance = COST_MAINTENANCE_YEARLY
    pv_revenue = revenue / ((1 + DISCOUNT_RATE) ** i)
    pv_maintenance = maintenance / ((1 + DISCOUNT_RATE) ** i)
    
    net_annual_pv = pv_revenue - pv_maintenance
    cumulative_net_cashflow += net_annual_pv
    cum_econ_val += pv_revenue
    
    lam = get_lambda(year)
    plot_data.append({"Year": year, "Lambda": lam, "Cumulative_Net_PV": cumulative_net_cashflow})

df_plot = pd.DataFrame(plot_data)


VAL_CULTURE = 241.45 * 1e6
VAL_ECON_TOTAL = cum_econ_val
VAL_SOCIAL_TOTAL = VAL_ECON_TOTAL + VAL_CULTURE


plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- 图表 1: 财务爬坡与风险剪刀差 (保持不变) ---
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in df_plot['Cumulative_Net_PV']]
bars = ax1.bar(df_plot['Year'], df_plot['Cumulative_Net_PV'] / 1e6, color=colors, alpha=0.8)
ax1_twin = ax1.twinx()
line, = ax1_twin.plot(df_plot['Year'], df_plot['Lambda'], color='#9b0000', linewidth=3, linestyle='--', marker='o', markersize=5)

ax1.annotate('Initial Investment Hole\n(-$550M CapEx)', xy=(2025, -550), xytext=(2028, -800),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
break_even_row = df_plot[df_plot['Cumulative_Net_PV'] > 0]
if not break_even_row.empty:
    be_year = break_even_row['Year'].min()
    ax1.annotate(f'Break-even Point\n(Year {be_year})', xy=(be_year, 0), xytext=(be_year - 1, 800),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),ha='center',va='center')

ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cumulative Net Project Value ($ Million)', fontsize=12, fontweight='bold')
ax1_twin.set_ylabel('Extreme Weather Frequency (Events/Year)', fontsize=12, color="#9b0000", fontweight='bold')
ax1.set_title('The "Funding Gap": Commercial Viability vs. Climate Risk', fontsize=14, fontweight='bold')
ax1.axhline(0, color='black', linewidth=1)
ax1_twin.grid(False)

legend_elements = [Patch(facecolor='#e74c3c', alpha=0.8, label='Deficit Phase (Gov Support Needed)'),
                   Patch(facecolor='#2ecc71', alpha=0.8, label='Surplus Phase (ROI Realized)'), line]
ax1.legend(handles=legend_elements, loc='upper left', labels=['Deficit Phase (Gov Support Needed)', 'Surplus Phase (ROI Realized)', 'Climate Risk Trend ($\lambda$)'])


# ✈️ 或 🧳 代表旅游，🏛️ 或 📜 代表文化遗产

labels_pie = ['Direct Economic Value\n(Tourism Revenue)', 
              'Intangible Cultural Value\n(Heritage & Symbolism)']
sizes_pie = [VAL_ECON_TOTAL, VAL_CULTURE]
colors_pie = ["#57b4f1", "#f9dc67"]
explode = (0, 0.1)
wedges, texts, autotexts = ax2.pie(sizes_pie, explode=explode, labels=labels_pie, colors=colors_pie,
                                   autopct='%1.1f%%', shadow=True, startangle=60,
                                   textprops=dict(color="black"), pctdistance=0.85)
centre_circle = plt.Circle((0,0),0.65,fc='white')
ax2.add_artist(centre_circle)
plt.setp(texts, size=12, weight="bold") 
plt.setp(autotexts, size=10, color="white", weight="bold")
ax2.text(0, 0, f"Total Social Value\n${VAL_SOCIAL_TOTAL/1e9:.2f} Billion", ha='center', va='center', fontsize=13, fontweight='bold')
ax2.set_title('Project Value Composition\n(Why Culture Matters)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()