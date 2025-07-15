import numpy as np
import lightgbm as lgb
import joblib
import time
from sklearn.model_selection import train_test_split
from lightgbm import record_evaluation
from colorama import init, Fore, Style

# Khởi động colorama
init()
RED = Fore.RED
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
RESET = Style.RESET_ALL

# Tiến trình hiển thị
def print_progress_bar(iteration, total, prefix='', suffix='', length=40):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

print(f"{YELLOW}[+] Đang tải dữ liệu huấn luyện...{RESET}")
X = np.load("X_combined.npy")
y = np.load("Y_combined.npy")
print(f"{GREEN}    → X shape: {X.shape}, y shape: {y.shape}{RESET}")

print(f"{YELLOW}[+] Đang chia dữ liệu train/test...{RESET}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"{GREEN}    → Train: {X_train.shape[0]} mẫu, Test: {X_test.shape[0]} mẫu{RESET}")

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.05,
    'verbose': -1,
    'is_unbalance': True
}

print(f"{YELLOW}[+] Đang huấn luyện mô hình LightGBM...{RESET}")
num_round = 1000
evals_result = {}
callback_eval = record_evaluation(evals_result)

start_time = time.time()

def progress_callback(env):
    print_progress_bar(env.iteration + 1, num_round, prefix="Training", length=40)

model = lgb.train(
    params,
    train_data,
    num_boost_round=num_round,
    valid_sets=[train_data, test_data],
    valid_names=["Train", "Test"],
    callbacks=[progress_callback, callback_eval]  # không dùng early stopping
)

end_time = time.time()
print(f"\n{GREEN}Huấn luyện xong sau {end_time - start_time:.2f}s. Model đã lưu vào: combined_model.pkl{RESET}")
print(f"{GREEN}Số lượng cây thực tế trong mô hình: {model.num_trees()}{RESET}")

# Lưu model
joblib.dump(model, "combined_model.pkl")

# Xuất sang dạng text giống Ember
model_str = model.model_to_string()
with open("combined_model.txt", "w", encoding="utf-8") as f:
    f.write(model_str)
print(f"{YELLOW}[+] Đã ghi mô hình dưới dạng text vào: combined_model.txt{RESET}")
