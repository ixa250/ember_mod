import os
import numpy as np
import ember
from tqdm import tqdm  # Thanh tiến trình

extractor = ember.PEFeatureExtractor()
X = []
folder = "benign_real/"

files = [f for f in os.listdir(folder) if f.lower().endswith(".exe")]

print(f"[+] Đang trích xuất đặc trưng từ {len(files)} file .exe...")

for fname in tqdm(files, desc="Trích xuất", unit="file"):
    path = os.path.join(folder, fname)
    try:
        with open(path, "rb") as f:
            bytez = f.read()
        vector = extractor.feature_vector(bytez)
        X.append(vector)
    except:
        continue  # Bỏ qua file lỗi

X_benign_real = np.array(X, dtype=np.float32)
y_benign_real = np.zeros(len(X_benign_real))  # label = 0

np.save("X_benign_real.npy", X_benign_real)
np.save("Y_benign_real.npy", y_benign_real)

print(f"\n✅ Đã trích xuất {len(X_benign_real)} vector. Lưu tại X_benign_real.npy & Y_benign_real.npy")
