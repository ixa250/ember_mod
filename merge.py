import numpy as np

X_old = np.load("X_combined.npy")
y_old = np.load("Y_combined.npy")

X_lief = np.load("X_benign_lief.npy")
y_lief = np.zeros(X_lief.shape[0], dtype=np.int32)  # benign = 0

X_new = np.concatenate([X_old, X_lief], axis=0)
y_new = np.concatenate([y_old, y_lief], axis=0)

np.save("X_combined_augmented.npy", X_new)
np.save("Y_combined_augmented.npy", y_new)

print(f"✅ Gộp thành công: {X_new.shape[0]} vector, lưu vào X_combined_augmented.npy & Y_combined_augmented.npy")
