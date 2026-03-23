import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

df = pd.read_csv('lung_cancer_chance.csv')

accuracies = []
results = []
full_fx_list = []

for i in range(1, 4):
    rs_shuffle = i * 24
    df_shuffled = df.sample(frac=1, random_state=rs_shuffle).reset_index(drop=True)
    
    X = df_shuffled.drop('lung_cancer_risk', axis=1)
    y = df_shuffled['lung_cancer_risk']
    
    rs_split = i * 99
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rs_split)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    accuracies.append(acc)
    
    intercept = model.intercept_[0]
    
    full_terms = [f"({coef:.4f}*{col})" for col, coef in zip(X.columns, model.coef_[0])]

    chunk_size = 5
    wrapped_terms = []
    for j in range(0, len(full_terms), chunk_size):
        wrapped_terms.append(" + ".join(full_terms[j:j+chunk_size]))
        
    fx_wrapped = f"Loop {i}:\nf(x) = {intercept:.4f}\n       + " + "\n       + ".join(wrapped_terms)
    full_fx_list.append(fx_wrapped)
    
    coef_first = model.coef_[0][0]
    col_first = X.columns[0]
    coef_last = model.coef_[0][-1]
    col_last = X.columns[-1]
    
    f_str_short = f"f(x) = {intercept:.4f} + ({coef_first:.4f}*{col_first}) + ... + ({coef_last:.4f}*{col_last})"
    
    results.append({
        'lan': f"Loop {i}",
        'dataset': f"Train ({X_train.shape[0]}), Test ({X_test.shape[0]})",
        'f_x': f_str_short,
        'acc': f"{acc:.2f}%"
    })

avg_acc = np.mean(accuracies)
results.append({
    'lan': "Average of Accuracy",
    'dataset': "-",
    'f_x': "-",
    'acc': f"{avg_acc:.2f}%"
})

print("-" * 100)
for fx in full_fx_list:
    print(fx)
    print("-" * 100)
print("\n")

w_lan = max(len("Iteration"), max(len(r['lan']) for r in results)) + 2
w_data = max(len("Dataset"), max(len(r['dataset']) for r in results)) + 2
w_fx = max(len("Summary function f(x)"), max(len(r['f_x']) for r in results)) + 2
w_acc = max(len("Accuracy"), max(len(r['acc']) for r in results)) + 2

def print_row(c1, c2, c3, c4):
    print(f"|{c1:^{w_lan}}|{c2:^{w_data}}|{c3:^{w_fx}}|{c4:^{w_acc}}|")

divider = f"+{'-'*w_lan}+{'-'*w_data}+{'-'*w_fx}+{'-'*w_acc}+"

print(divider)
print_row("Iteration", "Dataset", "Summary function f(x)", "Accuracy")
print(divider)

for r in results[:-1]:
    print_row(r['lan'], r['dataset'], r['f_x'], r['acc'])
print(divider)

print_row(results[-1]['lan'], results[-1]['dataset'], results[-1]['f_x'], results[-1]['acc'])
print(divider)
import matplotlib.pyplot as plt

# 1. Chuẩn bị dữ liệu cho biểu đồ
labels = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Average']
# Danh sách accuracies đã có 3 giá trị, ta thêm avg_acc vào cuối
plot_values = accuracies + [avg_acc] 

# 2. Cấu hình biểu đồ
plt.figure(figsize=(10, 6))
# Dùng màu xanh cho 3 lần lặp và màu cam cho giá trị trung bình để nổi bật
colors = ['#3498db', '#3498db', '#3498db', '#e67e22'] 
bars = plt.bar(labels, plot_values, color=colors)

# 3. Tùy chỉnh hiển thị (Quan trọng)
# Zoom trục Y để thấy rõ sự chênh lệch nhỏ giữa các lần lặp
plt.ylim(min(plot_values) - 20, 100.5) 
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Accuracy Table', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 4. Hiển thị con số cụ thể trên đầu mỗi cột
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}%', 
             ha='center', va='bottom', fontweight='bold')

# 5. Lưu hoặc hiển thị
plt.tight_layout()
plt.savefig('accuracy_chart.png') # Lưu thành file ảnh
plt.show() # Hiển thị lên màn hình