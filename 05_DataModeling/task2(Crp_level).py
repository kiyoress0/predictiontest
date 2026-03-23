import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # Thêm thư viện vẽ biểu đồ
import warnings

warnings.filterwarnings('ignore')

try:
    # 1. Load Data
    df = pd.read_csv('lung_cancer_chance.csv')
    df.columns = df.columns.str.strip().str.lower()

    # Cột cần loại bỏ khỏi features
    drop_cols = ['crp_level']
    target = 'lung_cancer_risk'

    # --- ĐỊNH NGHĨA NHÓM DỰA TRÊN CRP LEVEL ---
    crp_groups = {
        "f2 (Low CRP: 0-2)": df[df['crp_level'] <= 2].copy(),
        "f3 (Mid CRP: 3-9)": df[(df['crp_level'] >= 3) & (df['crp_level'] <= 9)].copy(),
        "f4 (High CRP: >=10)": df[df['crp_level'] >= 10].copy()
    }

    # Lưu trữ kết quả để tính trung bình
    all_metrics = {name: {'acc': [], 'size': len(data)} for name, data in crp_groups.items()}

    def get_formula_string(name, model, feature_names):
        """Create a series of equations f(x) using the normalized variable."""
        if model is None:
            return f"Model {name}: f(x) = Constant (Not enough classes to train)"
        
        intercept = model.intercept_[0]
        coefs = model.coef_[0]
        
        formula_parts = [f"{intercept:.4f}"]
        for coef, feat in zip(coefs, feature_names):
            sign = "+" if coef >= 0 else "-"
            formula_parts.append(f"{sign} ({abs(coef):.4f} * {feat}_scaled)")
        
        return f"Model {name}: f(x) = " + " ".join(formula_parts)

    def run_experiment_round(round_num):
        print(f"\n{'='*40} TESTING ROUND {round_num} (STANDARDIZED) {'='*40}")
        round_results = []
        round_formulas = []

        for name, data in crp_groups.items():
            n_samples = len(data)
            if n_samples < 5: continue

            # Tách Features và Target
            X_group = data.drop(columns=[target] + drop_cols)
            y_group = data[target]
            feature_names = X_group.columns.tolist()
            
            # Chia tập Train/Test ngẫu nhiên mỗi vòng
            test_size_val = 0.2 if n_samples < 1000 else 200
            X_train, X_test, y_train, y_test = train_test_split(
                X_group, y_group, test_size=test_size_val, shuffle=True
            )
            
            # --- THỰC HIỆN CHUẨN HÓA (SCALING) ---
            # Quan trọng: Fit trên Train và Transform cả Train/Test
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            unique_classes = np.unique(y_train)
            
            if len(unique_classes) < 2:
                # Trường hợp dữ liệu trong nhóm chỉ có 1 nhãn (ví dụ toàn 0 hoặc toàn 1)
                y_pred = np.full(shape=len(y_test), fill_value=unique_classes[0])
                acc = accuracy_score(y_test, y_pred)
                formula_str = get_formula_string(name, None, feature_names)
            else:
                model = LogisticRegression(max_iter=3000)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                formula_str = get_formula_string(name, model, feature_names)

            all_metrics[name]['acc'].append(acc)
            round_formulas.append(formula_str)
            round_results.append({
                'name': name,
                'size': n_samples,
                'train': len(X_train),
                'test': len(X_test),
                'acc': acc * 100
            })

        # --- IN CÔNG THỨC ---
        print("\nMATHEMATICAL EQUATION f(x) (Using normalized variables):")
        for f in round_formulas:
            print(f)
            print("-" * 20)

        # --- IN BẢNG CHI TIẾT VÒNG CHẠY ---
        print("\nDETAILED STATISTICS:")
        header = f"{'Nhóm Mô Hình':<25} | {'Size':<6} | {'Train':<6} | {'Test':<6}  | {'Accuracy (%)'}"
        print(header)
        print("-" * len(header))
        for res in round_results:
            print(f"{res['name']:<25} | {res['size']:<6} | {res['train']:<6} | {res['test']:<6}  | {res['acc']:>11.2f}%")

    # Thực thi 3 vòng để lấy số liệu trung bình
    for r in range(1, 4):
        run_experiment_round(r)

    # --- TỔNG KẾT TRUNG BÌNH CUỐI CÙNG ---
    print(f"\n{'='*30} TỔNG KẾT TRUNG BÌNH SAU 3 VÒNG {'='*30}")
    footer_head = f"{'Analysis Group CRP':<35} | {'Average accuracy (%)'}"
    print(footer_head)
    print("-" * len(footer_head))
    for name, metrics in all_metrics.items():
        if metrics['acc']:
            avg_acc = np.mean(metrics['acc']) * 100
            print(f"{name:<35} | {avg_acc:>18.2f}%")

    # ==========================================
    # --- VẼ BIỂU ĐỒ TRỰC QUAN HÓA ACCURACY ---
    # ==========================================
    groups = list(all_metrics.keys())
    
    # Khởi tạo mảng lưu dữ liệu
    round1, round2, round3, avgs = [], [], [], []
    
    for name in groups:
        metrics = all_metrics[name]
        # Kiểm tra xem nhóm có đủ 3 vòng test không
        if len(metrics['acc']) >= 3:
            round1.append(metrics['acc'][0] * 100)
            round2.append(metrics['acc'][1] * 100)
            round3.append(metrics['acc'][2] * 100)
            avgs.append(np.mean(metrics['acc']) * 100)
        else:
            round1.append(0)
            round2.append(0)
            round3.append(0)
            avgs.append(0)

    # Cấu hình biểu đồ
    x = np.arange(len(groups))  # Vị trí các nhóm trên trục X
    width = 0.2                 # Độ rộng của mỗi cột

    fig, ax = plt.subplots(figsize=(10, 6))

    # Vẽ 4 cột cho mỗi nhóm (Lần 1, 2, 3 và Trung bình)
    rects1 = ax.bar(x - 1.5*width, round1, width, label='Round 1', color='#4c72b0')
    rects2 = ax.bar(x - 0.5*width, round2, width, label='Round 2', color='#dd8452')
    rects3 = ax.bar(x + 0.5*width, round3, width, label='Round 3', color='#55a868')
    rects4 = ax.bar(x + 1.5*width, avgs, width, label='Average', color='#c44e52')

    # Trang trí biểu đồ
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy per CRP Group across 3 Rounds and Average', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontweight='bold')
    ax.set_ylim(0, 115) # Mở rộng trục Y để không che mất nhãn số

    # Chuyển legend xuống dưới
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)

    # Hàm hỗ trợ hiển thị số liệu % trên đỉnh mỗi cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0: # Chỉ in số nếu > 0
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # Nâng lên 3 points
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.tight_layout()
    plt.show() # Lệnh hiển thị biểu đồ

except Exception as e:
    print(f"Lỗi hệ thống: {e}")