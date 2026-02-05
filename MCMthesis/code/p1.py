import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel("ath.xlsx")  
countries = df.iloc[:, 0]             
X_raw = df.iloc[:, 1:].values         
features = df.columns[1:].tolist()   

scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)   

std_mean_score = X_std.mean(axis=1)   

pca = PCA(n_components=2)
pca_scores = pca.fit_transform(X_std)

pc1_scores = pca_scores[:, 0]   
pc2_scores = pca_scores[:, 1]   


result_df = pd.DataFrame({
    'Country': countries,
    'PC1_Score': pc1_scores,
    'PC2_Score': pc2_scores,
    'Std_Mean_Score': std_mean_score 
})

result_df = result_df.sort_values('PC1_Score', ascending=True).reset_index(drop=True)

plt.figure(figsize=(6, 6))
plt.scatter(result_df['PC1_Score'], result_df['PC2_Score'], alpha=0.5)
for i in range(len(result_df)):
    plt.text(result_df['PC1_Score'][i], result_df['PC2_Score'][i], ' ' , fontsize=9, ha='right')

plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.grid(True)
plt.show()

print("\n方差贡献率：")
print(f"PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.2%}")