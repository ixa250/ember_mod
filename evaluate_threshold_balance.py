import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample
from colorama import init, Fore, Style
init()

GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
CYAN = Fore.CYAN
RESET = Style.RESET_ALL

print(f"{YELLOW}[+] Đang tải dữ liệu...{RESET}")
X = np.load("X_combined_augmented.npy").astype(np.float16)
y = np.load("Y_combined_augmented.npy")

# Downsample như khi train
X_benign = X[y == 0]
X_malware = X[y == 1]
y_benign = y[y == 0]
y_malware = y[y == 1]

X_malware_down, y_malware_down = resample(X_malware, y_malware, n_samples=100_000, random_state=42)
X_benign_down, y_benign_down = resample(X_benign, y_benign, n_samples=80_000, random_state=42)

X_bal = np.concatenate((X_benign_down, X_malware_down), axis=0)
y_bal = np.concatenate((y_benign_down, y_malware_down), axis=0)

indices = np.random.permutation(len(X_bal))
X_bal = X_bal[indices]
y_bal = y_bal[indices]

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

print(f"{YELLOW}[+] Đang load mô hình...{RESET}")
model = joblib.load("balanced_model.pkl")

print(f"{YELLOW}[+] Dự đoán xác suất trên tập test...{RESET}")
y_prob = model.predict(X_test)

best_f1 = 0
best_threshold = 0

print(f"\n{CYAN}==== ĐÁNH GIÁ MÔ HÌNH THEO NGƯỠNG ===={RESET}")
print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
print("-" * 50)

for thresh in np.arange(0.3, 0.91, 0.05):
    y_pred = (y_prob > thresh).astype(int)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh
        print(f"{GREEN}{thresh:<10.2f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}{RESET}")
    else:
        print(f"{thresh:<10.2f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")

print(f"\n{GREEN}→ Ngưỡng tốt nhất: {best_threshold:.2f} (F1 = {best_f1:.4f}){RESET}")
