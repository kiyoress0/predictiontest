import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc dữ liệu
df = pd.read_csv('lung_cancer_chance.csv')

# 2. Tính ma trận tương quan Spearman
corr_spearman = df.corr(method='spearman')

# 3. Cài đặt kích thước và vẽ biểu đồ Heatmap
plt.figure(figsize=(22, 18)) 

# Dùng seaborn để vẽ heatmap
sns.heatmap(corr_spearman, 
            annot=True,            # Hiển thị con số bên trong ô
            fmt=".2f",             # Lấy 2 chữ số thập phân (vd: 0.81)
            cmap='RdBu_r',         # Bảng màu Đỏ (Đồng biến) - Xanh (Nghịch biến)
            vmin=-1, vmax=1,       # Giới hạn thang đo từ -1 đến 1
            linewidths=0.5,        # Kẻ viền trắng giữa các ô cho dễ nhìn
            annot_kws={"size": 8}) # Chỉnh cỡ chữ số bên trong ô

# 4. Trang trí tiêu đề và trục
plt.title('Spearman Correlation Matrix - Full Dataset', fontsize=22, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)

plt.tight_layout()
plt.show()