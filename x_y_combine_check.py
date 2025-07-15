import numpy as np

# Nếu dùng .npy
X_combined = np.load("X_combined.npy")
Y_combined = np.load("Y_combined.npy")

print("X_combined.shape:", X_combined.shape)
print("Y_combined.shape:", Y_combined.shape)
print("Phân bố nhãn:", np.bincount(Y_combined))
