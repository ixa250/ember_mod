import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from colorama import init, Fore, Style

# Màu terminal
init()
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
RESET = Style.RESET_ALL

print(f"{YELLOW}[+] Đang tải dữ liệu...{RESET}")
X = np.load("X_combined_augmented.npy").astype(np.float16)
y = np.load("Y_combined_augmented.npy")

# Downsample đúng như lúc train
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

print(f"{YELLOW}[+] Đang dự đoán với threshold = 0.55...{RESET}")
y_prob = model.predict(X_test)
y_pred = (y_prob > 0.55).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

# In kết quả 
print(f"\n{GREEN}====[ KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ]===={RESET}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC Score: {auc:.4f}")

print(f"\n{GREEN}====[ CONFUSION MATRIX ]===={RESET}")
print(f"{YELLOW}         Predicted{RESET}")
print(f"        [  0     1 ]")
print(f"Actual 0 [{cm[0][0]:5d} {cm[0][1]:5d}]")
print(f"       1 [{cm[1][0]:5d} {cm[1][1]:5d}]")
