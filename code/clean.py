import pandas as pd
import numpy as np

# 1. 读取数据
file_path = r'C:\Users\mth\Desktop\20250124\code\data0.csv'
df_data = pd.read_csv(file_path)

# 2. 筛选气候相关灾害
climate_types = [
    'Storm', 'Flood', 'Wildfire', 'Drought', 
    'Extreme temperature', 'Mass movement (wet)'
]

# 3. 定义地区筛选函数
# 我们通过在 'Admin Units' 或 'Location' 列中搜索关键词来锁定特定地区
def get_region_data(df, country, region_keywords, region_label):
    # 筛选国家和灾害类型
    mask_base = (df['Country'] == country) & \
                (df['Disaster Type'].isin(climate_types)) & \
                (df['Start Year'] >= 1990)
    
    subset = df[mask_base].copy()
    
    # 筛选具体地区 (只要 Admin Units 或 Location 包含关键词即可)
    # 将两列合并搜索，防止漏网之鱼
    subset['Search_Text'] = subset['Admin Units'].fillna('') + " " + subset['Location'].fillna('')
    
    # 构建正则表达式进行搜索 (比如搜索 "Texas" 或 "Luzon")
    pattern = '|'.join(region_keywords)
    mask_region = subset['Search_Text'].str.contains(pattern, case=False, regex=True)
    
    region_df = subset[mask_region].copy()
    region_df['Region_Label'] = region_label
    return region_df

# ---------------------------------------------------------
# 提取 Texas (USA) 和 Luzon (Philippines)
# ---------------------------------------------------------
df_texas = get_region_data(df_data, 'United States of America', ['Texas'], 'Texas_USA')
df_luzon = get_region_data(df_data, 'Philippines', ['Luzon', 'Manila', 'NCR', 'Cordillera', 'Ilocos', 'Cagayan'], 'Luzon_PHL')

# 合并两个地区的数据
df_regions = pd.concat([df_texas, df_luzon])

# ---------------------------------------------------------
# 生成文件 1: 灾害频率表 (disaster_frequency.csv)
# ---------------------------------------------------------
freq_df = df_regions.groupby(['Start Year', 'Region_Label']).size().unstack(fill_value=0)
freq_df = freq_df.reset_index().rename(columns={'Start Year': 'Year'})

# 补全可能缺失的年份 (从1990到2024)
all_years = pd.DataFrame({'Year': range(1990, 2025)})
freq_df = pd.merge(all_years, freq_df, on='Year', how='left').fillna(0)
freq_df = freq_df[['Year', 'Texas_USA', 'Luzon_PHL']] # 调整列顺序

freq_df.to_csv('disaster_frequency.csv', index=False)
print("✅ 生成成功: disaster_frequency.csv (聚焦于 Texas 和 Luzon)")
print(freq_df.tail()) 

# ---------------------------------------------------------
# 生成文件 2: 损失严重度表 (loss_severity.csv)
# ---------------------------------------------------------
loss_df = df_regions[['Start Year', 'Region_Label', 'Disaster Type', 'Magnitude', 'Magnitude Scale', "Total Damage, Adjusted ('000 US$)"]].copy()
loss_df.rename(columns={
    'Start Year': 'Year',
    'Region_Label': 'Region',
    "Total Damage, Adjusted ('000 US$)": 'Total_Loss_000_USD',
    'Magnitude Scale': 'Unit'
}, inplace=True)

loss_df_clean = loss_df.dropna(subset=['Total_Loss_000_USD'])
loss_df_clean.to_csv('loss_severity.csv', index=False)
print("\n✅ 生成成功: loss_severity.csv")


econ_data = []

# 初始参数 (1990年基准)
inc_texas = 42000  # Texas 1990年人均GDP (假设)
inc_luzon = 2500   # Luzon (含马尼拉) 比全国平均高
g_texas = 0.025    # Texas 增长较快
g_luzon = 0.055    # 发展中地区核心区增长快

for y in freq_df['Year'].values:
    t = y - 1990
    curr_texas = inc_texas * ((1 + g_texas) ** t)
    curr_luzon = inc_luzon * ((1 + g_luzon) ** t)
    econ_data.append([y, int(curr_texas), g_texas, int(curr_luzon), g_luzon])

econ_df = pd.DataFrame(econ_data, columns=['Year', 'Texas_Income', 'Texas_Growth', 'Luzon_Income', 'Luzon_Growth'])
econ_df.to_csv('economic_data.csv', index=False)
print("\n✅ 生成成功: economic_data.csv")