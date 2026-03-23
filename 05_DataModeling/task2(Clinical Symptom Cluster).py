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

    # Các cột dùng để chia nhóm
    drop_cols = ['xray_abnormal', 'chronic_cough', 'shortness_of_breath']
    target = 'lung_cancer_risk'

    # --- ĐỊNH NGHĨA CÁC NHÓM LÂM SÀNG ---
    groups_dict = {
        "f2 (No Symps & Xray Normal)": df[(df['xray_abnormal'] == 0) & 
                                         (df['chronic_cough'] == 0) & 
                                         (df['shortness_of_breath'] == 0)].copy(),
        
        "f3 (Symps but Xray Normal)": df[(df['xray_abnormal'] == 0) & 
                                        ((df['chronic_cough'] == 1) | (df['shortness_of_breath'] == 1))].copy(),
        
        "f4 (Symptoms & Xray Abnormal)": df[(df['xray_abnormal'] == 1) & 
                                           ((df['chronic_cough'] == 1) | (df['shortness_of_breath'] == 1))].copy()
    }

    # Cấu trúc lưu trữ để tính trung bình sau cùng
    all_results = {name: {'acc': [], 'size': len(data)} for name, data in groups_dict.items()}

    def get_formula_string(name, model, feature_names):
        if model is None:
            return f"Model {name}: f(x) = Constant"
        
        intercept = model.intercept_[0]
        coefs = model.coef_[0]
        
        formula_parts = [f"{intercept:.4f}"]
        for coef, feat in zip(coefs, feature_names):
            sign = "+" if coef >= 0 else "-"
            formula_parts.append(f"{sign} ({abs(coef):.4f} * {feat}_scaled)")
        
        return f"Model {name}: f(x) = " + " ".join(formula_parts)

    def run_experiment_round(r_idx):
        print(f"\n{'='*40} TESTING ROUND {r_idx} (STANDARDIZED) {'='*40}")
        round_data_list = []
        round_formulas = []

        for name, data in groups_dict.items():
            if len(data) < 5: continue
                
            X_g = data.drop(columns=[target] + drop_cols)
            y_g = data[target]
            feature_names = X_g.columns.tolist()
            
            # Chia Train/Test ngẫu nhiên
            X_train, X_test, y_train, y_test = train_test_split(X_g, y_g, test_size=0.1, shuffle=True)
            
            # --- CHUẨN HÓA (SCALING) ---
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                acc = accuracy_score(y_test, np.full(len(y_test), unique_classes[0]))
                formula_str = get_formula_string(name, None, feature_names)
            else:
                model = LogisticRegression(max_iter=3000)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                formula_str = get_formula_string(name, model, feature_names)

            all_results[name]['acc'].append(acc)
            round_formulas.append(formula_str)
            round_data_list.append({'name': name, 'n': len(data), 'acc': acc*100})

        print("\nMATHEMATICAL EQUATIONS f(x):")
        for f_text in round_formulas:
            print(f_text + "\n" + "-"*20)

        print("\nDETAILED ROUND STATISTICS " + str(r_idx) + ":")
        for r in round_data_list:
            print(f"{r['name']:<30} | Size: {r['n']:<5} | Accuracy: {r['acc']:.2f}%")

    # Thực hiện 3 vòng chạy
    for r in range(1, 4):
        run_experiment_round(r)

    # --- TÍNH VÀ IN KẾT QUẢ TRUNG BÌNH CUỐI CÙNG ---
    print(f"\n{'='*30} AVERAGE OVERVIEW AFTER 3 ROUNDS {'='*30}")
    header = f"{'Clinical Group':<35} | {'Average accuracy (%)'}"
    print(header)
    print("-" * len(header))
    
    for name, result in all_results.items():
        if result['acc']:
            avg_acc = np.mean(result['acc']) * 100
            print(f"{name:<35} | {avg_acc:>18.2f}%")

    # ==========================================
    # --- VẼ BIỂU ĐỒ TRỰC QUAN HÓA ACCURACY ---
    # ==========================================
    groups = list(all_results.keys())
    
    # Khởi tạo mảng lưu dữ liệu (xử lý an toàn nếu nhóm thiếu data)
    round1, round2, round3, avgs = [], [], [], []
    
    for name in groups:
        metrics = all_results[name]
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

    fig, ax = plt.subplots(figsize=(11, 6)) # Tăng chiều rộng một chút cho tên nhóm dài

    # Vẽ 4 cột cho mỗi nhóm (Lần 1, 2, 3 và Trung bình)
    rects1 = ax.bar(x - 1.5*width, round1, width, label='Round 1', color='#4c72b0')
    rects2 = ax.bar(x - 0.5*width, round2, width, label='Round 2', color='#dd8452')
    rects3 = ax.bar(x + 0.5*width, round3, width, label='Round 3', color='#55a868')
    rects4 = ax.bar(x + 1.5*width, avgs, width, label='Average', color='#c44e52')

    # Trang trí biểu đồ
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy per Clinical Group across 3 Rounds and Average', fontweight='bold')
    ax.set_xticks(x)
    
    # Do tên nhóm lâm sàng hơi dài, mình thêm '\n' để tự động xuống dòng giúp hiển thị đẹp hơn
    formatted_labels = [label.replace(' &', '\n&').replace(' but', '\nbut') for label in groups]
    ax.set_xticklabels(formatted_labels, fontweight='bold', fontsize=9)
    
    ax.set_ylim(0, 115) # Mở rộng trục Y để không che mất nhãn số

    # Chuyển legend xuống dưới để nhìn rõ hơn
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=4)

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

    # Tự động điều chỉnh layout để không bị cắt chữ
    plt.tight_layout()
    plt.show() # Lệnh hiển thị biểu đồ

except Exception as e:
    print(f"Lỗi hệ thống: {e}")