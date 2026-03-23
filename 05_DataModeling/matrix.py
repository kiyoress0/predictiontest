import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data và chuẩn hóa tên cột
df = pd.read_csv('lung_cancer_chance.csv')
df.columns = df.columns.str.strip().str.lower()

# 2. Tính ma trận tương quan cho toàn bộ DataFrame
corr_matrix = df.corr()

# 3. Chỉ lọc ra cột tương quan của 'lung_cancer_risk' 
# và sắp xếp giảm dần để thấy yếu tố nào ảnh hưởng mạnh nhất
target_corr = corr_matrix[['lung_cancer_risk']].sort_values(by='lung_cancer_risk', ascending=False)

# 4. Vẽ Heatmap dạng cột đơn (Single Column Heatmap)
plt.figure(figsize=(6, 10))
sns.heatmap(target_corr, 
            annot=True, 
            fmt=".3f", 
            cmap='YlOrRd', 
            xticklabels=['Lung Cancer Risk'], 
            yticklabels=target_corr.index)

plt.title("Correlation with Lung Cancer Risk")
plt.show()

# 5. In ra text
print("Hệ số tương quan so với Lung Cancer Risk:")
print(target_corr)



