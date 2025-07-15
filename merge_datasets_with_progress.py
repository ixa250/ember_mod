import numpy as np

def print_progress_bar(iteration, total, prefix='', suffix='', length=40):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

print("[+] Đang tải EMBER...")
X_ember = np.memmap("train_features/ember2018/X_train.dat", dtype=np.float32, mode="r").reshape(-1, 2381)
y_ember_raw = np.memmap("train_features/ember2018/y_train.dat", dtype=np.uint8, mode="r")
y_ember = np.array(y_ember_raw[:X_ember.shape[0]])

# ⚠️ CHUẨN HÓA nhãn: 0 giữ nguyên, các nhãn khác -> 1
y_ember_binary = np.where(y_ember == 0, 0, 1)

print(f"    → Tải EMBER xong: {X_ember.shape[0]} mẫu")

print("[+] Đang tải malware thật...")
X_real = np.load("X.npy")
y_real = np.load("y.npy")
print(f"    → Tải malware thật xong: {X_real.shape[0]} mẫu")

print("[+] Đang gộp dữ liệu...")
X_combined = np.empty((X_ember.shape[0] + X_real.shape[0], 2381), dtype=np.float32)
y_combined = np.empty((y_ember_binary.shape[0] + y_real.shape[0],), dtype=np.uint8)

# Gộp EMBER
for i in range(X_ember.shape[0]):
    X_combined[i] = X_ember[i]
    y_combined[i] = y_ember_binary[i]
    if i % 100000 == 0 or i == X_ember.shape[0] - 1:
        print_progress_bar(i + 1, X_ember.shape[0], prefix="Thêm từ EMBER ", suffix="sussces")

# Gộp malware thật
for i in range(X_real.shape[0]):
    X_combined[X_ember.shape[0] + i] = X_real[i]
    y_combined[y_ember_binary.shape[0] + i] = y_real[i]
    if i % 500 == 0 or i == X_real.shape[0] - 1:
        print_progress_bar(i + 1, X_real.shape[0], prefix="Thêm malware   ", suffix="sussces")

print("\n[+] Đang lưu lại X_combined.npy và Y_combined.npy...")
np.save("X_combined.npy", X_combined)
np.save("Y_combined.npy", y_combined)

print(f"[✅] Gộp xong! Tổng số mẫu: {X_combined.shape[0]}")
print("📊 Phân bố nhãn:", np.bincount(y_combined))
