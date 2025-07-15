import numpy as np
import lightgbm as lgb
import joblib
import time
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from colorama import init, Fore, Style

init()
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
RESET = Style.RESET_ALL

print(f"{YELLOW}[+] Đang tải dữ liệu gốc...{RESET}")
X = np.load("X_combined.npy").astype(np.float16)  # giảm 1 nửa RAM
y = np.load("Y_combined.npy")

# Tách benign và malware
X_benign = X[y == 0]
y_benign = y[y == 0]
X_malware = X[y == 1]
y_malware = y[y == 1]

# Giảm malware còn 100k, benign còn 80k
X_malware_down, y_malware_down = resample(X_malware, y_malware,
                                          n_samples=100_000,
                                          random_state=42)
X_benign_down, y_benign_down = resample(X_benign, y_benign,
                                        n_samples=80_000,
                                        random_state=42)

# Ghép lại
X_bal = np.concatenate((X_benign_down, X_malware_down), axis=0)
y_bal = np.concatenate((y_benign_down, y_malware_down), axis=0)

# Shuffle bằng chỉ số
indices = np.random.permutation(len(X_bal))
X_bal = X_bal[indices]
y_bal = y_bal[indices]

print(f"{GREEN}    → Dataset sau cân bằng: {X_bal.shape}, Malware: {np.sum(y_bal==1)}, Benign: {np.sum(y_bal==0)}{RESET}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_bin': 255,
    'min_child_samples': 50,
    'verbose': -1
}

print(f"{YELLOW}[+] Đang huấn luyện mô hình LightGBM...{RESET}")
start_time = time.time()
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=["Train", "Test"],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)
end_time = time.time()
print(f"{GREEN}Huấn luyện xong sau {end_time - start_time:.2f}s. Số cây: {model.best_iteration}{RESET}")

joblib.dump(model, "balanced_model.pkl")
print(f"{YELLOW}[+] Đã lưu model vào: balanced_model.pkl{RESET}")
