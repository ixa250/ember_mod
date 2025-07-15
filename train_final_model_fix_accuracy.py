import numpy as np
import lightgbm as lgb
import joblib
import time
from sklearn.model_selection import train_test_split
from lightgbm import record_evaluation
from colorama import init, Fore, Style
import os
import sys

init()
RED = Fore.RED
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
RESET = Style.RESET_ALL

def print_stage(message):
    print(f"{YELLOW}[+] {message}...{RESET}")

def print_success(message):
    print(f"{GREEN}    → {message}{RESET}")

def print_progress_bar(iteration, total, prefix='', suffix='', length=40):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def simulate_progress(task_name, duration=1.5):
    print_progress_bar(0, 100, prefix=task_name, suffix='', length=30)
    for i in range(1, 101):
        time.sleep(duration / 100)
        print_progress_bar(i, 100, prefix=task_name, suffix='', length=30)

# Bước 1: Load dữ liệu
print_stage("Đang tải dữ liệu huấn luyện")
simulate_progress("Tải X")
X = np.load("X_combined_augmented.npy")
simulate_progress("Tải y")
y = np.load("Y_combined_augmented.npy")
print_success(f"X shape: {X.shape}, y shape: {y.shape}")

# Bước 2: Chia train/test
print_stage("Chia dữ liệu thành tập huấn luyện và kiểm thử")
simulate_progress("Chia dữ liệu")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print_success(f"Train: {X_train.shape[0]} mẫu, Test: {X_test.shape[0]} mẫu")

# Bước 3: Tính tỉ lệ dương/âm
neg = np.sum(y_train == 0)
pos = np.sum(y_train == 1)
weight_ratio = neg / pos
print_success(f"Số mẫu âm: {neg}, dương: {pos}, tỉ lệ weight: {weight_ratio:.2f}")

# Bước 4: Tạo dataset LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Bước 5: Thiết lập tham số
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'scale_pos_weight': weight_ratio,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_bin': 255,
    'min_child_samples': 50,
    'verbose': -1
}

# Bước 6: Huấn luyện mô hình
print_stage("Đang huấn luyện mô hình LightGBM")
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
    callbacks=[progress_callback, callback_eval, lgb.early_stopping(stopping_rounds=50)]
)

end_time = time.time()
print_success(f"Huấn luyện xong sau {end_time - start_time:.2f}s. Số cây: {model.best_iteration}")

# Bước 7: Lưu mô hình
joblib.dump(model, "combined_model.pkl")
print_success("Đã lưu mô hình vào combined_model.pkl")

# Ghi dưới dạng text
model_str = model.model_to_string()
with open("combined_model.txt", "w", encoding="utf-8") as f:
    f.write(model_str)
print_success("Đã ghi mô hình dưới dạng text vào: combined_model.txt")
