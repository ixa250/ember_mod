import numpy as np

# Đọc dữ liệu vector đã tạo
X = np.load("X_combined_augmented.npy")
y = np.load("Y_combined_augmented.npy")

# In ra kích thước (shape) của 2 mảng
print("X shape:", X.shape)  # (số sample, số feature)
print("Y shape:", y.shape)
