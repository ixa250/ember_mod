import numpy as np

# Load EMBER 2018 dataset (.dat)
X_ember = np.memmap("train_features/ember2018/X_train.dat", dtype=np.float32, mode="r").reshape(-1, 2381)

# Load y_ember đúng theo số lượng mẫu X_ember
y_ember_raw = np.memmap("train_features/ember2018/y_train.dat", dtype=np.uint8, mode="r")
y_ember = np.array(y_ember_raw[:X_ember.shape[0]])  # Trích đúng số nhãn

# Load real malware dataset (.npy)
X_real = np.load("X.npy")
y_real = np.load("y.npy")  # đổi tên cho rõ ràng nếu cần

# Gộp dữ liệu
X_combined = np.vstack([X_ember, X_real])
y_combined = np.concatenate([y_ember, y_real])

# Kiểm tra đồng bộ số mẫu
assert X_combined.shape[0] == y_combined.shape[0], f"Sai số lượng mẫu: X={X_combined.shape[0]}, y={y_combined.shape[0]}"

# Lưu thành file mới
np.save("X_combined.npy", X_combined)
np.save("Y_combined.npy", y_combined)

print(f"Gộp thành công! Tổng số mẫu: {X_combined.shape[0]}")
